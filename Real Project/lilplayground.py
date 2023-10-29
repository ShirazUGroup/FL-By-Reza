'''
a = (1,2)

print(a[0])
'''

'''
import itertools


models = ['a','b','c','d','e','f']

number_of_epochs_values = [5 * (i+1) for i in range(len(models))]

number_of_epochs_values = [5 * (i+1) for i in range(len(models))]
pruning_ratio_values = [i / 10.0 for i in range(11)]

hyperparameter_combinations = itertools.product(number_of_epochs_values, pruning_ratio_values)


for num_epochs, pruning_ratio in hyperparameter_combinations:
    print(f"n: {num_epochs}, p: {pruning_ratio}, model: {models[int((num_epochs/5)-1)]}")
'''
'''
number_of_epochs_values = [5, 10, 15, 20]
pruning_ratio_values = [i / 10.0 for i in range(11)]

        # Generate all combinations of hyperparameters
hyperparameter_combinations = itertools.product(number_of_epochs_values, pruning_ratio_values)

for n,p in hyperparameter_combinations:
    print(f"n: {n}, p: {p}")
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
import copy
from torch.utils.data import SubsetRandomSampler, DataLoader


#HyperParameters

max_iterations = 20
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

            if (i+1) % 100 == 0:
                print ("Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}"
                    .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

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

        print(f'{str} device 0 : {100 * correct / total}')



 


# Calculate the number of samples per client
epoch = 50
epoch_setter = EpochTuning()
pruning=Pruning()
device = torch.device("mps")



# Create a DataLoader for the subset corresponding to the current device
#subset_train = torch.utils.data.Subset(train_dataset, device_indices)
#train_loader = torch.utils.data.DataLoader(subset_train, batch_size=100, shuffle=False)

num_samples = 10000

indices = torch.randperm(len(train_dataset))[:num_samples]
sampler = SubsetRandomSampler(indices)
batch_size = 100
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, sampler=sampler)
model = ResNet(ResidualBlock, [2, 2, 2]).to(device)
models,times = train_model(model,train_loader,device,epoch)
for m in models:
    evaluate_model("accuracy: ",m,device)
for t in times:
    print(f"time: {t}")    
        




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