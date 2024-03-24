import torch
import numpy as np
from mmvae.data import MappedCellCensusDataLoader
from torch.nn.functional import cosine_similarity

class SparseDataset():
    def __init__(self, npz_path, device):
        self.device = device
        self.chunk_file = npz_path
        self.dataloader = self.configure_dataloader()
        self.data = self.dataloader.dataset.data.to_dense()
        self.mutant_data = None
        self.healthy_data = None
        
    def configure_dataloader(self):
        return MappedCellCensusDataLoader(
            batch_size=32,
            device=self.device,
            file_path=self.chunk_file,
            load_all=True
        )
        
    def set_healthy_indices(self, healthy_indices_list, mutant_indices_list):
        print(f"healthy_indices_list: {len(healthy_indices_list)}")
        print(f"mutant_indices_list: {len(mutant_indices_list)}")
        
        self.healthy_data = self.data[healthy_indices_list]
        self.mutant_data = self.data[mutant_indices_list]
        
        self.data = None
        pass

    def __len__(self):
        return self.data.shape[0]

    def __get_healthy_item__(self, idx):
        # Return the dense vector of the item
        return self.healthy_data[idx]
    
    def __get_mutant_item__(self, idx):
        # Return the dense vector of the item
        return self.mutant_data[idx]
    
    def find_nearest(self, cell_vector):
        cell_vector = torch.tensor(cell_vector, dtype=torch.float32, device=self.device)
        max_similarity = -1
        nearest_neighbor_idx = -1
        
        # Compute cosine similarity between cell_vector and each cell in the dataset
        for idx, data in enumerate(self.healthy_data):
            similarity = cosine_similarity(cell_vector.unsqueeze(0), data)
            if similarity > max_similarity:
                max_similarity = similarity
                nearest_neighbor_idx = idx

        return nearest_neighbor_idx, max_similarity.item()


def main(device):
    # Example usage
    npz_path = '/active/debruinz_project/human_data/python_data/human_chunk_10.npz'  # Path to your .npz file
    dataset = SparseDataset(npz_path, device=device)
    print("Finished loading datset")

    # cell_vector = np.random.rand(dataset.dataloader.dataset.data.shape[1])  # Example vector
    cell_vector = dataset.__getitem__(0)
    print(f"cell_vector: {cell_vector}")
    nearest_neighbor_idx, similarity = dataset.find_nearest(cell_vector)

    print(f"Nearest neighbor index: {nearest_neighbor_idx}, Similarity: {similarity}")


if __name__ == "__main__":
    CUDA = True
    device = "cuda" if torch.cuda.is_available() and CUDA else "cpu"
    main(device)