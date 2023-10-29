from mpi4py import MPI
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

class Server:

    def __init__(self,Comm):

        self.comm = Comm
    pruning_mask_list=[]    
    #-------------------------------------------------
    def receive_weights_from_clients_and_calculate_average_of_one_layer(self,param,num_clients,pruning_mask_list,index_layer):
            tensor_data = param.data
            tensor_data_temp = np.zeros_like(param.data)
            
            for i in range(1,num_clients+1):
                if pruning_mask_list[i-1][index_layer]!=0:
                # Clone the tensor data to avoid modifying the original weights
                    self.comm.Recv(tensor_data_temp, source=i)
                    #print(f"temp is: {np.mean(tensor_data_temp)} from {i}")
                    tensor_data += tensor_data_temp
            tensor_data =tensor_data/ num_clients
            param.data=tensor_data

    def receive_weights_of_all_layers_and_update_model(self,model,num_clients,pruning_mask_list):
        j=0
        for name, param in model.named_parameters():
            self.receive_weights_from_clients_and_calculate_average_of_one_layer(param,num_clients,pruning_mask_list,j)
            j+=1

    #---------------------------------
    def send_aggregated_weights_of_one_layer_to_clients(self,param,num_clients):
        tensor_data = param.data.clone()  # Clone the tensor data to avoid modifying the original weights
        for i in range(1,num_clients+1):
            self.comm.Send(tensor_data.numpy(), dest=i)
    def send_aggregated_weights_of_all_layers_to_clients(self,model,num_clients):
        for name, param in model.named_parameters():
            self.send_aggregated_weights_of_one_layer_to_clients(param,num_clients)
    #---------------------------------------------------
    
    def receive_pruning_masks_from_clients(self,layer_size,num_clients):
            pruning_mask_list=[]
            for i in range(1,num_clients+1):
                prunning_mask_temp = np.zeros(layer_size)
                # Clone the tensor data to avoid modifying the original weights
                self.comm.Recv(prunning_mask_temp, source=i)
                #print(f"temp is: {np.mean(tensor_data_temp)} from {i}")
                pruning_mask_list.append(prunning_mask_temp)
            #print("\n\n-------------------\n mask list:",pruning_mask_list)
            return pruning_mask_list
            