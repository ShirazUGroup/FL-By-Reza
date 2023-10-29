from mpi4py import MPI
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from client import Client
from server import Server
from epochtuning import EpochTuning
import random
import time
from  pruning import Pruning
from optimizer import Optimizer
import copy

#HyperParameters

max_iterations = 20
layer_count=47

num_samples = 100

loss_weight = 0.9
energy_weight = 0.1
min_epoch = 20
max_epoch = 50

###############################################


# Image preprocessing modules
transform = transforms.Compose([
    transforms.Pad(4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32),
    transforms.ToTensor()])

# CIFAR-10 dataset
train_dataset = torchvision.datasets.CIFAR10(root='../../../data/',train=True,transform=transform,download=True)

test_dataset = torchvision.datasets.CIFAR10(root='../../../data/',train=False,transform=transforms.ToTensor())

# Data loader
#train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=100, shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=100, shuffle=False)
'''
subset_indices_train = range(100)
subset_train = torch.utils.data.Subset(train_dataset, subset_indices_train)
train_loader = torch.utils.data.DataLoader(subset_train, batch_size=1, shuffle=False)

subset_indices_test = range(20)
subset_test = torch.utils.data.Subset(test_dataset, subset_indices_test)
test_loader = torch.utils.data.DataLoader(subset_test, batch_size=1, shuffle=False)
'''
# 3x3 convolution

#--------------------------------------------------------------------

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
    t1 = time.process_time()
    # Loss and optimizer
    num_epochs = epoch
    learning_rate = 0.001
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    checkpoints =[]
    times = []

    # For updating learning rate
    def update_lr(optimizer, lr):
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    # Train the model
    total_step = len(train_loader)
    curr_lr = learning_rate
    total_time = 0
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

            if (i+1) % 25 == 0:
                print ("Device: {} Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}"
                    .format(rank,epoch+1, num_epochs, i+1, total_step, loss.item()))

        # Decay learning rate
        if (epoch+1) % 20 == 0:
            curr_lr /= 3
            update_lr(optimizer, curr_lr)
        if (epoch+1) % 5 == 0:
            model_copy = copy.deepcopy(model)
            checkpoints.append(model_copy)
        if (epoch+1) % 10 == 0:
            times.append(time.process_time()-t1)

    return checkpoints,times

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

#---------------------------------------------------
def calculate_mean_with_varying_divisor(numbers):

    results = []

    for i, num in enumerate(numbers):
        divisor = (i + 1) * 10
        result = num / divisor
        results.append(result)

    if len(results) > 0:
        mean_result = sum(results) / len(results)
    else:
        mean_result = 0
    return mean_result
#--------------------------------------------------

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
    
    #evaluate_model("before agg accuracy",model,device)

    for iter in range(max_iterations):
        pruning_mask_list=server.receive_pruning_masks_from_clients(layer_count,num_clients)
        server.receive_weights_of_all_layers_and_update_model(model,num_clients,pruning_mask_list)
        server.send_aggregated_weights_of_all_layers_to_clients(model,num_clients)
        #evaluate_model(f"after agg accuracy in iter {iter}",model,device)

#--------------------------------------------------
 
def client():

    # Calculate the number of samples per client
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    num_clients = size - 1
    samples_per_client = num_samples // num_clients
    num_epochs = 20
    pruning_ratio = 0
    rank = comm.Get_rank()
    client = Client(comm)
    epoch_setter = EpochTuning()
    pruning=Pruning()
    device = torch.device("cpu")
    optimizer = Optimizer()
    # Randomly choose samples for the current device
    random.seed(rank)  # Set seed for reproducibility based on rank
    all_indices = list(range(num_samples))
    random.shuffle(all_indices)
    device_indices = all_indices[(rank-1) * samples_per_client:(rank) * samples_per_client]
    #print(f"device {rank} indices length: {len(device_indices)}")

    # Create a DataLoader for the subset corresponding to the current device
    subset_train = torch.utils.data.Subset(train_dataset, device_indices)
    train_loader = torch.utils.data.DataLoader(subset_train, batch_size=100, shuffle=False)


    for iter in range(max_iterations):

        print(f"iter: {iter}, device: {rank}, selected num_epochs: {num_epochs}, selected pruning_ratio: {pruning_ratio}")

        if (iter + 1) % 5 == 0:

            globalModel = copy.deepcopy(model)

            #print(f"device {rank} started training in iter {iter} with {num_epochs} epochs")
            train_start_time = time.time()
            models,times = train_model(model,train_loader,device,max_epoch)
            train_end_time = time.time()
            evaluate_model(f"before agg accuracy in iter {iter}",model,device)
            time_per_epoch = calculate_mean_with_varying_divisor(times)
            #print(f"time per epoch from device {rank}: {time_per_epoch}")
            result_pair = optimizer.find_optimal_hyperparameters(globalModel,models,time_per_epoch,energy_weight,loss_weight,rank*100,700,min_epoch,max_epoch,device)

            pruning_ratio = result_pair[1]
            num_epochs = result_pair[0]

            pruning_mask=pruning.generate_mask(model,result_pair[1])

            pruning.apply_pruning(model,pruning_mask,globalModel)

            #pruning.sort_layers_by_l2norm(model,rank)
            #print(f"device {rank} pruning mask is : {pruning_mask}")
            client.send_pruning_mask_to_server(pruning_mask)
            client.send_weights_of_all_layers_to_server(model,pruning_mask)  
            client.receive_aggregated_weights_of_all_layers(model)
            #evaluate_model(f"after agg accuracy in iter {iter} ",model,device)
            training_time = train_end_time-train_start_time
            epoch = epoch_setter.set_epoch(training_time)

        else:
            if iter == 0:
                model = ResNet(ResidualBlock, [2, 2, 2]).to(device)
                globalModel = model

            #print(f"device {rank} started training in iter {iter} with {num_epochs} epochs")
            train_start_time = time.time()
            train_model(model,train_loader,device,num_epochs)
            train_end_time = time.time()
            #print(f"training time took : {train_end_time-train_start_time} and the type is {type(train_end_time-train_start_time)}")
            #evaluate_model(f"before agg accuracy in iter {iter}",model,device)

            pruning_mask=pruning.generate_mask(model,pruning_ratio)

            #mask = pruning.generate_mask(model,1)
            pruning.apply_pruning(model,pruning_mask,globalModel)

            #pruning.sort_layers_by_l2norm(model,rank)
            #print(f"device {rank} pruning mask is : {pruning_mask}")
            client.send_pruning_mask_to_server(pruning_mask)
            client.send_weights_of_all_layers_to_server(model,pruning_mask)  
            client.receive_aggregated_weights_of_all_layers(model)
            #evaluate_model(f"after agg accuracy in iter {iter} ",model,device)
            #training_time = train_end_time-train_start_time
            #epoch = epoch_setter.set_epoch(training_time)
        #print(iter)

#--------------------------------------------------------------------------------------        

if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if rank == 0:
        server()
    else:
        client()

'''

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import time
import random
from pruning import Pruning
import numpy as np

from pruning import Pruning

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
        accuracy = 100 * correct / total
    return accuracy

# Device configuration
device = torch.device('mps')

# Hyper-parameters
num_epochs = 1
learning_rate = 0.001

# Image preprocessing modules
transform = transforms.Compose([
    transforms.Pad(4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32),
    transforms.ToTensor()])

# CIFAR-10 dataset
train_dataset = torchvision.datasets.CIFAR10(root='../../data/',
                                             train=True,
                                             transform=transform,
                                             download=True)

test_dataset = torchvision.datasets.CIFAR10(root='../../data/',
                                            train=False,
                                            transform=transforms.ToTensor())


# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=100, shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=100, shuffle=False)

results = np.full((50, 10), -1)

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

#model = ResNet(ResidualBlock, [2, 2, 2]).to(device)
bestModel = torch.load("ResNet18-MAC").to(device)
randModel = ResNet(ResidualBlock, [2, 2, 2]).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(randModel.parameters(), lr=learning_rate)



# For updating learning rate
def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# Train the model
############################## profiling 

for n in range(1,51):
  #subset_indices = random.sample(range(1, 50000), 50)
  #subset = torch.utils.data.Subset(train_dataset, subset_indices)
  #train_loader = torch.utils.data.DataLoader(subset, batch_size=1, shuffle=False)
  total_time = 0
  total_step = len(train_loader)
  curr_lr = learning_rate
  for epoch in range(n):
      for i, (images, labels) in enumerate(train_loader):
          images = images.to(device)
          labels = labels.to(device)
          t1 = time.process_time()
          # Forward pass
          outputs = model(images)
          loss = criterion(outputs, labels)

          # Backward and optimize
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()
          t2 = time.process_time()
          total_time += (t2-t1)
          #if (i+1) % 100 == 0:
          #print ("Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}".format(epoch+1, num_epochs, i+1, total_step, loss.item()))

      # Decay learning rate
      if (epoch+1) % 20 == 0:
          curr_lr /= 3
          update_lr(optimizer, curr_lr)

  backupModel = model
  # Test the model
  for i in range(1, 11):
    pruning = Pruning()

    ratio = i / 10.0
    mask = pruning.generate_mask(model,ratio)
    pruning.apply_pruning(model,mask)
    accuracy = evaluate_model("",model,device)
    results[n-1][i-1] = accuracy
    print('Accuracy of the model on the test images: {} %'.format(accuracy))
    model = backupModel
  
  print('number of random samples= 50 '+'number of epochs= '+str(n)+' total time elapsed in milliseconds: {}'.format(total_time*1000))
  for layer in model.children():
   if hasattr(layer, 'reset_parameters'):
       layer.reset_parameters()
  model = ResNet(ResidualBlock, [2, 2, 2]).to(device)


pruning = Pruning()

mask = pruning.generate_mask(bestModel,0.1)

pruning.apply_pruning(bestModel,mask,randModel)

print(f"Accuracy {evaluate_model('',bestModel,device)}")

file_path = 'my_array.npy'

# Save the array to disk
np.save(file_path, results)
print(torch.cuda.is_available())

# Save the model checkpoint
torch.save(randModel.state_dict(), 'resnet.ckpt')
'''

'''
from mpi4py import MPI
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from client import Client
from server import Server
from epochtuning import EpochTuning
import random
import time
from  pruning import Pruning
from optimizer import Optimizer

#Model Training
max_iterations = 5
layer_count=47
###############################################


# Image preprocessing modules
transform = transforms.Compose([
    transforms.Pad(4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32),
    transforms.ToTensor()])

# CIFAR-10 dataset
train_dataset = torchvision.datasets.CIFAR10(root='../../../data/',train=True,transform=transform,download=True)

test_dataset = torchvision.datasets.CIFAR10(root='../../../data/',train=False,transform=transforms.ToTensor())

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=100, shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=100, shuffle=False)


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

        print(f'{str} accuracy : {100 * correct / total}')

#----------------------------------------------------

def count_parameters(model):
    """
    Count the total number of parameters in a PyTorch model.

    Args:
        model (torch.nn.Module): The PyTorch model.

    Returns:
        int: The total number of parameters in the model.
    """
    return sum(p.numel() for p in model.parameters())

 
epoch=2
epoch_setter = EpochTuning()
pruning=Pruning()
device = torch.device("cuda")

model = torch.load("ResNet18-MAC").to(device)
#model = ResNet(ResidualBlock, [2, 2, 2]).to(device)
#train_start_time = time.time()
#train_model(model,train_loader,device,epoch)
#train_end_time = time.time()
#print(f"training time took : {train_end_time-train_start_time} and the type is {type(train_end_time-train_start_time)}")
#evaluate_model(f"before agg accuracy in iter {iter}",model,device)
#pruning_mask=pruning.generate_mask(model,0.7)
#pruning.apply_pruning(model,pruning_mask)
#evaluate_model(f"after agg accuracy in iter {iter} ",model,device)
#optimizer = Optimizer()
#print(f"before pruning size: {count_parameters(model)} after pruning size: {optimizer.calculate_model_size_with_mask(model,pruning_mask)}")




#print(f"device {rank} pruning mask is : {pruning_mask}")
#training_time = train_end_time-train_start_time
#epoch = epoch_setter.set_epoch(training_time)
'''


'''
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import time
import random
from optimizer import Optimizer
from pruning import Pruning

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
num_epochs = 1
learning_rate = 0.001

# Image preprocessing modules
transform = transforms.Compose([
    transforms.Pad(4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32),
    transforms.ToTensor()])

# CIFAR-10 dataset
train_dataset = torchvision.datasets.CIFAR10(root='../../../data/',
                                             train=True,
                                             transform=transform,
                                             download=True)

test_dataset = torchvision.datasets.CIFAR10(root='../../../data/',
                                            train=False,
                                            transform=transforms.ToTensor())


# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=100, shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=100, shuffle=False)

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

model = ResNet(ResidualBlock, [2, 2, 2]).to(device)


# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# For updating learning rate
def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# Train the model
total_times = []
for x in range(10):
  #subset_indices = random.sample(range(1, 50000), 50)
  #subset = torch.utils.data.Subset(train_dataset, subset_indices)
  #train_loader = torch.utils.data.DataLoader(subset, batch_size=1, shuffle=False)
  total_time = 0
  total_step = len(train_loader)
  curr_lr = learning_rate
  for epoch in range(50):
      for i, (images, labels) in enumerate(train_loader):
          images = images.to(device)
          labels = labels.to(device)
          t1 = time.process_time()
          # Forward pass
          outputs = model(images)
          loss = criterion(outputs, labels)

          # Backward and optimize
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()
          t2 = time.process_time()
          total_time += (t2-t1)
          if (i+1) % 100 == 0:
            print ("Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}".format(epoch+1, num_epochs, i+1, total_step, loss.item()))

      # Decay learning rate
      if (epoch+1) % 20 == 0:
          curr_lr /= 3
          update_lr(optimizer, curr_lr)

  # Test the model
  model.eval()
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

      print('Accuracy of the model on the test images: {} %'.format(100 * correct / total))
  print('iteration ' + str(x) + 'all samples, '+'number of epochs= '+str(50)+' total time elapsed in milliseconds: {}'.format(total_time*1000))
  total_times.append(total_time)

  for layer in model.children():
   if hasattr(layer, 'reset_parameters'):
       layer.reset_parameters()

print(torch.cuda.is_available())

# Save the model checkpoint
torch.save(model.state_dict(), 'resnet.ckpt')
for item in total_times:
    print(item)
'''