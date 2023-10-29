import itertools
from pruning import Pruning
import copy
import torch
import torchvision
import time

test_dataset = torchvision.datasets.CIFAR10(root='../../../data/',train=False,transform=torchvision.transforms.ToTensor())
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=100, shuffle=False)

pruning = Pruning()

class Optimizer:


    def calculate_comp_energy(self,num_epochs,fixed_energy_per_epoch) :

        return num_epochs*fixed_energy_per_epoch

    def calculate_comm_energy(self,model,pruning_ratio):

        mask = pruning.generate_mask(model,pruning_ratio)
        return (pruning.calculate_model_size_with_mask(model,mask)/1000)

    def evaluate_model(self,str,model, device):
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
    

    

    def find_optimal_hyperparameters(self,globalModel,models,time_per_epoch,energy_weight,loss_weight,fixed_energy_per_epoch, max_time, min_epoch, max_epoch,device):

        pruning = Pruning()


        best_hyperparameters = None
        best_objective = float('inf')

        best_total_energy=0
        best_reduction_in_loss = 0

        # Define the possible values for number_of_epochs and pruning_ratio
        number_of_epochs_values = [5 * (i+1) for i in range(len(models))]
        pruning_ratio_values = [i / 10.0 for i in range(6)]

        max_reduction_in_loss = 87

        #max_total_energy = (max_epoch*fixed_energy_per_epoch) + ((pruning.calculate_model_size_with_mask(globalModel,pruning.generate_mask(globalModel,0)))/1000)

        max_total_energy = self.calculate_comp_energy(max_epoch,fixed_energy_per_epoch) + self.calculate_comm_energy(globalModel,0)

        # Generate all combinations of hyperparameters
        hyperparameter_combinations = itertools.product(number_of_epochs_values, pruning_ratio_values)

        for num_epochs, pruning_ratio in hyperparameter_combinations:

            #computation_energy = num_epochs * fixed_energy_per_epoch

            computation_energy = self.calculate_comp_energy(num_epochs,fixed_energy_per_epoch)

            model = copy.deepcopy(models[int((num_epochs/5)-1)])

            pruning_mask = pruning.generate_mask(model, pruning_ratio)
            
            #communication_energy = (pruning.calculate_model_size_with_mask(model, pruning_mask))/1000

            communication_energy = self.calculate_comm_energy(model,pruning_ratio)
            

            # Calculate the total energy for the current hyperparameters
            total_energy = computation_energy + communication_energy

            print(f"comp_energy: {computation_energy}, comm_energy: {communication_energy}, total_energy: {total_energy},max_total_energy: {max_total_energy}")

            pruning.apply_pruning(model,pruning_mask,globalModel)
            reduction_in_loss = 87 - (self.evaluate_model("",model,device))

            total_time = num_epochs*time_per_epoch

            # Check if the current hyperparameters satisfy the constraints
            if min_epoch <= num_epochs <= max_epoch and total_time <= max_time:
                # Calculate the objective function (weighted sum)
                objective = (energy_weight * (total_energy/max_total_energy)) + (loss_weight * (reduction_in_loss/max_reduction_in_loss))
                print(f"non_normalized_energy: {total_energy}, normalized_energy: {(energy_weight * (total_energy/max_total_energy))}, non_norm_reduc_in_loss: {reduction_in_loss}, norm_reduc_in_loss: {(loss_weight * (reduction_in_loss/max_reduction_in_loss))}")
                # Update the best hyperparameters if this is a better solution
                if objective < best_objective:
                    best_hyperparameters = (num_epochs, pruning_ratio)
                    best_objective = objective
                    best_total_energy = total_energy
                    best_reduction_in_loss = reduction_in_loss
        print(f"total_energy wrt optimized num_epochs and pruning_ratio: {best_total_energy}, reduction_in_loss: {best_reduction_in_loss}")
        return best_hyperparameters

# Example usage:
'''
best_hyperparameters = find_optimal_hyperparameters(
    pruning=Pruning(),
    evaluate_model=evaluate_model_function,  # Define your evaluate_model function
    max_time=your_max_time,
    min_epoch=your_min_epoch,
    max_epoch=your_max_epoch
)

print("Best Hyperparameters (number_of_epochs, pruning_ratio):", best_hyperparameters)
'''