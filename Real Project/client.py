from mpi4py import MPI
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

class Client:

    def __init__(self,Comm):
         self.comm = Comm
         
    #-----------------------------------------------------   
        
    def send_weights_of_one_layer_to_server(self,param):
        tensor_data = param.data.clone()  # Clone the tensor data to avoid modifying the original weights
        self.comm.Send(tensor_data.numpy(), dest=0)
    def send_weights_of_all_layers_to_server(self,model,mask):
        i=0
        for name, param in model.named_parameters():
            if mask[i]!=0:
                self.send_weights_of_one_layer_to_server(param)
            i+=1
            #print("\n\n\n +++++++++++++++++++++++ \n size of model ",i)
    #--------------------------------------------------------------------


    def receive_aggregated_weights_from_server(self,param):
            tensor_data = param.data
            
            self.comm.Recv(tensor_data, source=0)
                #print(f"temp is: {np.mean(tensor_data_temp)} from {i}")
            
            param.data=tensor_data

    def receive_aggregated_weights_of_all_layers(self,model):
        for name, param in model.named_parameters():
            self.receive_aggregated_weights_from_server(param)

    #------------------------------------------------------
    def send_pruning_mask_to_server(self,mask):
        self.comm.Send(mask, dest=0)