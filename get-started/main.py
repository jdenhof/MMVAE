import torch
# from mmvae.trainers import HumanVAETrainer
from datetime import datetime
# from csvFileLoader import CSVFileLoader
from mmvae.finetuners.nearest import SparseDataset
from mmvae.finetuners.finetune import PreTrained_Model
# import numpy as np
import mmvae.finetuners.dataset as fd
import csv


def main(device):
    dataset : SparseDataset
    ft_trainer : PreTrained_Model
    
    csv_file = '/home/howlanjo/dev/MMVAE/csv_data/top_100_values' + datetime.now().strftime("%Y%m%d-%H%M%S") + '.csv'
    
    print("Starting now")
    
    ft_trainer = PreTrained_Model()
    ft_trainer.load_pretrained_model()
    
    # # Define any hyperparameters
    # batch_size = 32
    
    # # Create trainer instance
    # trainer = HumanVAETrainer(
    #     batch_size, 
    #     device,
    #     log_dir='/home/howlanjo/logs/' + datetime.now().strftime("%Y%m%d-%H%M%S") + "FINE_TUNE",
    #     snapshot_path="/home/howlanjo/dev/MMVAE/snapshots/" + datetime.now().strftime("%Y%m%d-%H%M%S") + "FINE_TUNE" ,
    # )
    # # Train model with number of epochs
    # trainer.train(epochs = 2, load_snapshot=False)
    
    dataset = fd.create_knn_dataset()
    
    print(f"len: {len(dataset.mutant_data)}")
    
    for i in range(len(dataset.mutant_data)):
        cell_vector = 0    
        cell_vector = dataset.__get_mutant_item__(i)
        nearest_neighbor_idx, similarity, sim_list = dataset.find_nearest(cell_vector)
        
        
        # sim_list.sort(reverse=True)
        sim_list.sort(key=lambda x: x[0], reverse=True)
        # Step 2: Slice the top 100 values
        top_100 = sim_list[:100]

        # Step 3: Write the top 100 values to a CSV file
        with open(csv_file, 'a', newline='') as file:
            writer = csv.writer(file)
            # Writing each value in its own row
            for value in top_100:
                writer.writerow([i, value[0], value[1]])
            
            # print("Starting Training")
            # for i in range(100):
            #     ft_trainer.model.train_trace_complete(dataset.__get_healthy_item__(nearest_neighbor_idx), cell_vector, i)

    print("All done")
    
if __name__ == "__main__":
    CUDA = False
    device = "cuda" if torch.cuda.is_available() and CUDA else "cpu"
    main(device)
