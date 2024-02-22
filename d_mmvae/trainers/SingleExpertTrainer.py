import torch
import torch.nn.functional as F
import d_mmvae.trainers.utils as utils
import d_mmvae.models.SingleExpertVAE as SingleExpertVAE
from d_mmvae.trainers.trainer import BaseTrainer
from d_mmvae.data import MultiModalLoader, CellCensusDataLoader

class SingleExpertTrainer(BaseTrainer):
    """
    Trainer class designed for MMVAE model using MutliModalLoader.
    """

    model: SingleExpertVAE.Model
    dataloader: MultiModalLoader

    def __init__(self, batch_size, *args, **kwargs):
        self.batch_size = batch_size # Defined before super().__init__ as configure_* is called on __init__
        super(SingleExpertTrainer, self).__init__(*args, **kwargs)
        self.model.to(self.device)
        # self.expert_class_indices = [i for i in range(len(self.model.experts)) ]
        self.annealing_steps = 50

    def configure_dataloader(self):
        #expert = CellCensusDataLoader('expert', directory_path="/active/debruinz_project/CellCensus_3M", masks=['3m_human_chunk*'], batch_size=self.batch_size, num_workers=1)
        expert = CellCensusDataLoader('expert', directory_path="/active/debruinz_project/CellCensus_3M", masks=['3m_human_chunk_1*'], batch_size=self.batch_size, num_workers=1)
        return MultiModalLoader(expert)

    def configure_model(self):
        return SingleExpertVAE.configure_model()
    
    def configure_optimizers(self) -> dict[str, torch.optim.Optimizer]:
        return {
            'encoder': torch.optim.Adam(self.model.expert.encoder.parameters(), lr=0.0001),
            'decoder': torch.optim.Adam(self.model.expert.decoder.parameters(), lr=0.0001),
            'solo_vae': torch.optim.Adam(self.model.solo_vae.parameters(), lr=.00001)
        }

    def train_epoch(self, epoch):
        old_train_data = None
        for iteration, (data, _) in enumerate(self.dataloader):
            print("Starting Iteration", iteration, flush=True)
            train_data = data.to(self.device)
            
            # Zero All Gradients
            self.optimizers['solo_vae'].zero_grad()
            self.optimizers['encoder'].zero_grad()
            self.optimizers['decoder'].zero_grad()
                                                        
            # Forward Pass Over Entire Model
            re_param, mu, var, decoded = self.model(train_data)
            
            print(f"mu: {mu.mean()}")
            print(f"var: {var.mean()}")
        
            recon_loss = F.l1_loss(decoded, train_data.to_dense())
            
            # Shared VAE Loss
            kl_div = -0.5 * torch.sum(1 + var 
                                      - mu**2 
                                      - torch.exp(var), 
                                      axis=1) # sum over latent dimension
            
            kl_loss = kl_div.mean() # average over batch dimension
            
            #kl_loss = utils.kl_divergence(mu, var)
            kl_weight = 1 #min(1.0, epoch / self.annealing_steps)
            loss = recon_loss + (kl_loss * kl_weight)
            loss.backward()

            self.optimizers['solo_vae'].step()
            self.optimizers[f'encoder'].step()
            self.optimizers[f'decoder'].step()
            
            if iteration > 0:
                try:
                    baseline = F.l1_loss(train_data.to_dense(), old_train_data.to_dense())
                    self.writer.add_scalar('Baseline', baseline.item(), iteration)
                except:
                    pass
            
            self.writer.add_scalar('Loss/KL', kl_loss.item(), iteration)
            self.writer.add_scalar('Loss/ReconstructionFromTrainingData', recon_loss.item(), iteration)
            self.writer.add_scalar('Loss/TotalLoss', loss.item(), iteration)
            
            self.writer.flush()
            
            old_train_data = train_data