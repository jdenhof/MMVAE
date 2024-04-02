import torch
from mmvae.finetuners.csvFileLoader import CSVFileLoader
from mmvae.finetuners.nearest import SparseDataset
import numpy as np


def create_knn_dataset(device = "cpu",
                       folder_path = '/active/debruinz_project/human_data/python_data', 
                       filename_pattern='chunk10_metadata.csv',
                       disease_healthy='normal', 
                       disease_mutant='cystic fibrosis', 
                       tissue_type='lung',
                       chunk_data_file = '/active/debruinz_project/human_data/python_data/human_chunk_10.npz'):
    
    # Create an instance of the CSVFileLoader
    healthy_loader = CSVFileLoader(folder_path = folder_path,
                        filename_pattern = filename_pattern,  
                        disease = disease_healthy,
                        tissue_type = tissue_type)
    
    # Create an instance of the CSVFileLoader
    sick_loader = CSVFileLoader(folder_path = folder_path,
                        filename_pattern = filename_pattern,
                        disease = disease_mutant,
                        tissue_type = tissue_type)
    
    print("Finished CSVFileLoader")

    # Load the CSV files and filter rows
    healthy_loader.load_files()
    sick_loader.load_files()
    
    print(f"Number of healthy samples: {len(healthy_loader.matching_rows)}")
    print(f"Number of sick samples: {len(sick_loader.matching_rows)}")
    
    # Example usage
    dataset = SparseDataset(chunk_data_file, device=device)
    print("Finished loading datset")
    
    dataset.set_healthy_indices(healthy_loader.matching_indices, sick_loader.matching_indices)
    
    return dataset
    