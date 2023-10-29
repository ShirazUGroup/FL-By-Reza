from mpi4py import MPI
import numpy as np
import torch
import torch.nn as nn
import torchvision
from client import Client
from server import Server
from epochtuning import EpochTuning
import random
import time
from  pruning import Pruning

#Model Training
max_iterations = 5
layer_count=47
###############################################


# Image preprocessing modules
transform = torchvision.transforms.Compose([
    torchvision.transforms.Pad(4),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.RandomCrop(32),
    torchvision.transforms.ToTensor()])

# CIFAR-10 dataset
train_dataset = torchvision.datasets.CIFAR10(root='../../../data/',train=True,transform=transform,download=True)

test_dataset = torchvision.datasets.CIFAR10(root='../../../data/',train=False,transform=torchvision.transforms.ToTensor())

# Data loader
#train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=100, shuffle=True)

#test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=100, shuffle=False)

subset_indices_train = range(100)
subset_train = torch.utils.data.Subset(train_dataset, subset_indices_train)
train_loader = torch.utils.data.DataLoader(subset_train, batch_size=1, shuffle=False)

subset_indices_test = range(20)
subset_test = torch.utils.data.Subset(test_dataset, subset_indices_test)
test_loader = torch.utils.data.DataLoader(subset_test, batch_size=1, shuffle=False)

# 3x3 convolution
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                    stride=stride, padding=1, bias=False)

# Residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

# ResNet
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 16
        self.conv = conv3x3(3, 16)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(block, 16, layers[0])
        self.layer2 = self.make_layer(block, 32, layers[1], 2)
        self.layer3 = self.make_layer(block, 64, layers[2], 2)
        self.avg_pool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64, num_classes)

    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                conv3x3(self.in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

#----------------------------------------------------
def train_model(model,train_loader,device,epoch):
    # Loss and optimizer
    num_epochs = epoch
    learning_rate = 0.001
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # For updating learning rate
    def update_lr(optimizer, lr):
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    # Train the model
    total_step = len(train_loader)
    curr_lr = learning_rate
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % 100 == 0:
                print ("Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}"
                    .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

        # Decay learning rate
        if (epoch+1) % 20 == 0:
            curr_lr /= 3
            update_lr(optimizer, curr_lr)


#----------------------------------------------------
def evaluate_model(str,model, device):
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(f'{str} device {rank} : {100 * correct / total}')

def server():
    
    device = torch.device("cpu")
    model = ResNet(ResidualBlock, [2, 2, 2]).to(device)
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    server = Server(comm)
    
    if size <= 1:
        print("This program requires at least 2 processes: 1 server and 1 or more clients.")
        return

    num_clients = size - 1
    
    evaluate_model("before agg accuracy",model,device)

    for iter in range(max_iterations):
        pruning_mask_list=server.receive_pruning_masks_from_clients(layer_count,num_clients)
        server.receive_weights_of_all_layers_and_update_model(model,num_clients,pruning_mask_list)
        server.send_aggregated_weights_of_all_layers_to_clients(model,num_clients)
        evaluate_model(f"after agg accuracy in iter {iter}",model,device)

 
def client():
    epoch=2
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    client = Client(comm)
    epoch_setter = EpochTuning()
    pruning=Pruning()
    device = torch.device("cpu")

    for iter in range(max_iterations):
        if iter==0:
            model = torch.load("ResNet18-MAC").to(device)
        #model = ResNet(ResidualBlock, [2, 2, 2]).to(device)
        print(f"device {rank} started training in iter {iter} with {epoch} epochs")
        train_start_time = time.time()
        train_model(model,train_loader,device,epoch)
        train_end_time = time.time()
        #print(f"training time took : {train_end_time-train_start_time} and the type is {type(train_end_time-train_start_time)}")
        evaluate_model(f"before agg accuracy in iter {iter}",model,device)
        pruning_mask=pruning.generate_mask(model,layer_count)
        pruning.sort_layers_by_l2norm(model,rank)
        #print(f"device {rank} pruning mask is : {pruning_mask}")
        client.send_pruning_mask_to_server(pruning_mask)
        client.send_weights_of_all_layers_to_server(model,pruning_mask)  
        client.receive_aggregated_weights_of_all_layers(model)
        evaluate_model(f"after agg accuracy in iter {iter} ",model,device)
        training_time = train_end_time-train_start_time
        epoch = epoch_setter.set_epoch(training_time)


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if rank == 0:
        server()
    else:
        client()