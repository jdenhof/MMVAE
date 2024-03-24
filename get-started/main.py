import torch
from mmvae.trainers import HumanVAETrainer
from datetime import datetime
from csvFileLoader import CSVFileLoader
from nearest import SparseDataset
import numpy as np

def main(device):
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
    
    # Create an instance of the CSVFileLoader
    healthy_loader = CSVFileLoader(folder_path='/active/debruinz_project/human_data/python_data',
                        filename_pattern='chunk10_metadata.csv',  # This will match any .csv file
                        disease='normal',
                        tissue_type='lung')
    
    # Create an instance of the CSVFileLoader
    sick_loader = CSVFileLoader(folder_path='/active/debruinz_project/human_data/python_data',
                        filename_pattern='chunk10_metadata.csv',  # This will match any .csv file
                        disease='cystic fibrosis',
                        tissue_type='lung')

    # Load the CSV files and filter rows
    healthy_loader.load_files()
    sick_loader.load_files()
    
    print(f"Number of healthy samples: {len(healthy_loader.matching_rows)}")
    print(f"Number of sick samples: {len(sick_loader.matching_rows)}")
    
    # Example usage
    npz_path = '/active/debruinz_project/human_data/python_data/human_chunk_10.npz'  # Path to your .npz file
    dataset = SparseDataset(npz_path, device=device)
    print("Finished loading datset")
    
    dataset.set_healthy_indices(healthy_loader.matching_indices, sick_loader.matching_indices)
    print(f"dataset.data shape: {dataset.healthy_data.shape}")
    
    # cell_vector = np.random.rand(dataset.dataloader.dataset.data.shape[1])  # Example vector
    
    cell_vector = 0    
    cell_vector = dataset.__get_mutant_item__(0)
    print(f"cell_vector: {cell_vector}")
    nearest_neighbor_idx, similarity = dataset.find_nearest(cell_vector)

    print(f"Nearest neighbor index: {nearest_neighbor_idx}, Similarity: {similarity}")
    
if __name__ == "__main__":
    CUDA = True
    device = "cuda" if torch.cuda.is_available() and CUDA else "cpu"
    main(device)
