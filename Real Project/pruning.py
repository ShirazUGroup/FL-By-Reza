import random
import numpy as np
import torch

ResNet18LayerCount = 47

class Pruning:


    def calculate_model_size_with_mask(self,model, mask):

        if len(mask) != len(list(model.parameters())):
            raise ValueError("The length of the mask should match the number of model parameters.")

        pruned_model_params = [p for i, p in enumerate(model.parameters()) if mask[i] == 1]
        pruned_model_size = sum(p.numel() for p in pruned_model_params)

        return pruned_model_size

    
    def sort_layers_by_l2norm(self, model, rank):

        layer_info = []

        for name, param in model.named_parameters():
            num_params = param.data.numel()
            l2_norm = np.linalg.norm(param.data.cpu().numpy())
            layer_info.append((name, num_params, l2_norm))
        
        sorted_indexes = np.argsort([-l2_norm for _, _, l2_norm in layer_info])
        weight_indexes = [idx for idx in sorted_indexes if 'bias' not in layer_info[idx][0]]

        for idx in weight_indexes:
            layer_name, num_params, l2_norm = layer_info[idx]
            #print(f"Device: {rank} Layer: {layer_name}, Index: {idx}, Parameters: {num_params}, L2 Norm: {l2_norm}")
        
        return weight_indexes        

    def generate_mask(self,model,ratio_of_layers_to_prune):

        weight_indexes = self.sort_layers_by_l2norm(model, 0)
        number_of_layers_to_prune = int(ratio_of_layers_to_prune*ResNet18LayerCount)
        #print(f"number of layers to prune: {number_of_layers_to_prune}")
        selected_indexes = weight_indexes[-number_of_layers_to_prune:]
        
        mask = np.ones(ResNet18LayerCount)
        if number_of_layers_to_prune != 0:
            mask[selected_indexes] = 0
        #print(f"ratio {number_of_layers_to_prune}, the mask {mask}")
        return mask
    '''
    def apply_pruning(self, model, mask, globalModel):

        for idx, param in enumerate(model.parameters()):
            if mask[idx] == 0:
                param.data = torch.ones_like(param.data)
    '''
    def apply_pruning(self, model, mask, globalModel):
        global_params = list(globalModel.parameters())
        for idx, param in enumerate(model.parameters()):
            if mask[idx] == 0:
                param.data = global_params[idx].data.clone()

