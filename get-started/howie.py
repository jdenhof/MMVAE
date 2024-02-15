import torch
from d_mmvae.trainers import SingleExpertTrainer
from datetime import datetime

def main(device):
    # Define any hyperparameters
    batch_size = 32
    # Create trainer instance
    trainer = SingleExpertTrainer(
        batch_size,
        device,
        log_dir='/home/howlanjo/logs/' + "TESTING_" + datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    # Train model with number of epochs
    trainer.train(epochs=1)

if __name__ == "__main__":
    CUDA = True
    device = "cuda" if torch.cuda.is_available() and CUDA else "cpu"
    main(device)
