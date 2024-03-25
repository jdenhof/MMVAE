import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import LRScheduler
import mmvae.trainers.utils as utils
import mmvae.models.HumanVAE as HumanVAE
from mmvae.trainers import HPBaseTrainer, BaseTrainerConfig
import torch.utils.tensorboard as tb

class HumanVAEConfig(BaseTrainerConfig):
    required_hparams = {
        # 'data_file_path': str,
        # 'metadata_file_path': str,
        # 'train_dataset_ratio': float,
        'data.train.directory' : str,
        'data.train.batch_size': int,
        'data.train.start_chunk': int,
        'data.train.end_chunk': int,
        'data.test.directory': str,
        'data.test.batch_size': int,
        'data.test.start_chunk': int,
        'data.test.end_chunk': int,
        'expert.encoder.optimizer.lr': float,
        'expert.decoder.optimizer.lr': float, 
        'shr_vae.optimizer.lr': float, 
        'kl_cyclic.warm_start': float, 
        'kl_cyclic.cycle_length': float, 
        'kl_cyclic.min_beta': float, 
        'kl_cyclic.max_beta': float 
    }
            
class HumanMetricTracker:
    def __init__(self, hparams, writer: tb.writer.SummaryWriter):
        super(HumanMetricTracker, self).__init__()
        self.hparams = hparams
        self.writer = writer
        self.train_metrics = utils.MetricTracker()
        self.test_metrics = utils.MetricTracker()
        
    def log_architecture(self, model):
        self.writer.add_text('Model Architecture', str(model))
        
    def log_trace_test_dataset_results(self):
        metrics = {}
        # Record average reconstruction loss over entire dataset
        metrics['Test/Loss/Reconstruction'] = self.test_metrics['recon_loss'] / self.test_metrics.iteration
        # Record average kl loss over entire dataset
        metrics['Test/Loss/KL'] = self.test_metrics['kl_loss'] / self.test_metrics.iteration
        # Write metrics and hparams to hparam file
        self.writer.add_hparams(dict(self.hparams), metrics, run_name=f"{self.hparams['tensorboard.run_name']}_hparams", global_step=self.hparams['epochs'])
    
    def log_trace_train_batch_results(self, kl_weight, batch_iteration, logging_interval=100):
        # Log KL weight for each batch iteration
        self.writer.add_scalar('Metric/KLWeight', kl_weight, batch_iteration)
        iteration = self.train_metrics.iteration
        if iteration == logging_interval:
            self.writer.add_scalar('Train/Loss/Recon', self.train_metrics['recon_loss'] / iteration, batch_iteration)
            self.writer.add_scalar('Train/Loss/Total', self.train_metrics['loss'] / iteration, batch_iteration)
            self.writer.add_scalar('Train/Loss/KL', self.train_metrics['kl_loss'] / iteration, batch_iteration)
            self.writer.add_scalar('Train/Metrics/L1Penalty', self.train_metrics['l1_penalty'] / iteration, batch_iteration)
            self.train_metrics.reset()
        
class HumanVAETrainer(HPBaseTrainer):
    
    model: HumanVAE.Model
    
    def __init__(self, _device: torch.device, _hparams: HumanVAEConfig):
        super(HumanVAETrainer, self).__init__(_device, _hparams)
        self.metric_tracker = HumanMetricTracker(self.hparams, self.writer)
        
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #                             Configuration                             #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    
    configure_model = HumanVAE.configure_model
    
    def configure_dataloader(self):
        import mmvae.data as md
    
        def generate_masks(name: str):
            return [f'human_chunk_{key}.npz' for key in range(self.hparams[f'data.{name}.start_chunk'], self.hparams[f'data.{name}.end_chunk'] + 1)]

        self.train_loader, self.test_loader = md.configure_multichunk_dataloaders(
            self.hparams['data.train.batch_size'],
            self.hparams['data.train.directory'],
            generate_masks('train'),
            self.hparams['data.test.batch_size'],
            self.hparams['data.test.directory'],
            generate_masks('test'),
            verbose=False,
        )
        
    def configure_optimizers(self):
        return {
            'expert.encoder': torch.optim.Adam(self.model.expert.encoder.parameters(), lr=self.hparams['expert.encoder.optimizer.lr']),
            'expert.decoder': torch.optim.Adam(self.model.expert.decoder.parameters(), lr=self.hparams['expert.decoder.optimizer.lr']),
            'shr_vae': torch.optim.Adam(self.model.shared_vae.parameters(), lr=self.hparams['shr_vae.optimizer.lr']),
        }
        
    def configure_schedulers(self) -> dict[str, LRScheduler]:
        return { 
                key: torch.optim.lr_scheduler.StepLR(
                    optimizer, 
                    step_size=self.hparams[f'{key}.optimizer.schedular.step_size'], 
                    gamma=self.hparams[f"{key}.optimizer.schedular.gamma"])
                for key, optimizer in self.optimizers.items() 
                if f'{key}.optimizer.schedular.step_size' in self.hparams 
                    and self.hparams[f'{key}.optimizer.schedular.step_size'] != "" 
                    and f'{key}.optimizer.schedular.gamma' in self.hparams
                    and self.hparams[f'{key}.optimizer.schedular.gamma'] != ""
            }
        
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #                          Trace Configuration                          #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    
    def get_kl_weight(self):
        warm_start = self.hparams['kl_cyclic.warm_start']
        if self.batch_iteration < warm_start:
            return 0
        return utils.cyclic_annealing(
            (self.batch_iteration - warm_start), 
            self.hparams['kl_cyclic.cycle_length'], 
            min_beta=self.hparams['kl_cyclic.min_beta'], 
            max_beta=self.hparams['kl_cyclic.max_beta']
        )
    
    def trace_expert_reconstruction(self, train_data: torch.Tensor, reduction="sum"):
        x_hat, mu, logvar, z = self.model(train_data)
        recon_loss = F.mse_loss(x_hat, train_data.to_dense(), reduction=reduction)
        kl_loss = utils.kl_divergence(mu, logvar, reduction=reduction)
        return x_hat, z, mu, logvar, recon_loss, kl_loss
    
    def log_non_zero_and_zero_reconstruction(self, inputs, targets):
        non_zero_mask = inputs != 0
        self.metrics['Test/Loss/NonZeroFeatureReconstruction'] += F.mse_loss(inputs[non_zero_mask], targets[non_zero_mask], reduction='sum') 
        zero_mask = ~non_zero_mask
        self.metrics['Test/Loss/ZeroFeatureReconstruction'] += F.mse_loss(inputs[zero_mask], targets[zero_mask], reduction='sum') 
    
    def trace_test_dataset(self, epoch):
        self.test_loader.seed(epoch)
        self.metric_tracker.test_metrics.reset()
        with torch.no_grad():
            self.model.eval()
            for test_data in self.test_loader:
                test_data = test_data.to(self.device)
                _, _, _, _, recon_loss, kl_loss = self.trace_expert_reconstruction(test_data, reduction='mean')
                recon_loss, kl_loss = recon_loss.item(), kl_loss.item()
                loss = recon_loss + kl_loss
                self.metric_tracker.test_metrics.update({
                    'recon_loss': recon_loss,
                    'kl_loss': kl_loss,
                    'loss': loss
                })
        self.metric_tracker.log_trace_test_dataset_results()
        
    def trace_train_batch(self, train_data: torch.Tensor):
        
        self.optimizers['shr_vae'].zero_grad()
        self.optimizers['expert.encoder'].zero_grad()
        self.optimizers['expert.decoder'].zero_grad()
        
        xhat, z, _, _, recon_loss, kl_loss = self.trace_expert_reconstruction(train_data)

        l1_penalty = torch.abs(z).sum()
        kl_weight = self.get_kl_weight()
        loss: torch.Tensor = recon_loss + (kl_weight* kl_loss) + (kl_weight * l1_penalty)
        
        loss.backward()
        
        self.optimizers['shr_vae'].step()
        self.optimizers['expert.encoder'].step()
        self.optimizers['expert.decoder'].step()
        
        self.metric_tracker.train_metrics.update({
            'recon_loss': recon_loss.detach().item() / xhat.numel(),
            'kl_loss':  kl_loss.detach().item() / z.numel(),
            'loss': loss.detach().item() / xhat.numel(),
            'l1_penalty': l1_penalty.item()
        })
        self.metric_tracker.log_trace_train_batch_results(kl_weight, self.batch_iteration)
    
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #                          Train Configuration                          #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    def train(self, epochs, load_snapshot=False):
        # Initialize batch iteration to -1 to signal no batch
        self.batch_iteration = -1
        # Training loop
        super().train(epochs, load_snapshot)
        # Shutdown multiprocessing dataloaders
        self.train_loader.shutdown()
        self.test_loader.shutdown()

    def train_epoch(self, epoch):
        # Set the seed for the training dataloader
        self.train_loader.seed(epoch)
        # Make sure model will record grad
        self.model.train()
        # Iterate over all samples in the trainig dataset
        for train_data in self.train_loader:
            # Signal batch receivied index at 0
            self.batch_iteration += 1
            # Send training data to gpu
            train_data = train_data.to(self.device)
            self.trace_train_batch(train_data)
            # Step for all schedulars
            for schedular in self.schedulers.values():
                schedular.step()
        self.trace_test_dataset(epoch)
        # Set new epoch completion 
        self.hparams['epochs'] = epoch
            
    