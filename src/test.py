import torch
import os

if __name__ == "__main__":
    from torch.distributed import init_process_group, destroy_process_group
    init_process_group(backend="nccl") # DDP Setup

    if not torch.cuda.is_available():
        devices = "cpu"
    else:
        devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    assert(devices is not None and devices == "cpu" or len(devices) > 0)
    print(f"Main running with device(s): {devices}")

    from torch.utils.data import IterableDataset
    class SparseDataset(IterableDataset):
        def __iter__(self):
            for _ in range(32):
                yield torch.rand(1, 256).to_sparse_csr()

    from torch import nn
    model = nn.Sequential(nn.Linear(256, 128), nn.Linear(128, 32))

    optimizer = torch.optim.Adam(model.parameters()) # TODO: Use distributed optimizer

    from torch.utils.data import DataLoader
    loader = DataLoader(
        SparseDataset(),
        num_workers=2,
        batch_size=None,
        shuffle=False,
        pin_memory=True, # pin_memory_device not set so default is cpu 
        prefetch_factor=2,
        persistent_workers=True,
    )

    local_rank = int(os.environ["LOCAL_RANK"])
    from torch.nn.parallel import DistributedDataParallel
    model = DistributedDataParallel(model.to(local_rank), device_ids=[local_rank])
    for _ in range(2):
        for source in loader:
            source = source.to(local_rank)
            optimizer.zero_grad()
            output, mean, logvar = model(source)
            recon_loss = torch.nn.CrossEntropyLoss()(output, source.to_dense())
            kld = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
            loss = recon_loss + kld
            loss.backward()
            optimizer.step()

    destroy_process_group()
    print("DONE")