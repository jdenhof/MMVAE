CHUNK_BUFFER_SIZE=5
CHUNK_DIRECTORY='/active/debruinz_project/tony_boos/csr_chunks/'
SAVE_EVERY=5
SNAPSHOT_PATH='./snapshots'
CHUNK_SIZE=285341
SEED=123456
NUM_SAMPLES=10
SAMPLE_SIZE=CHUNK_SIZE * NUM_SAMPLES
MODEL_PATH='/home/denhofja/D-MMVAE/src/Models.py'
NUM_WORKERS=5

#Just a thought but if we are already are giving up true random sampling within each chunk if we DDP across all 4 gpus we can hold the entire dataset split across the 4 gpus. Meaning we possibly could load in the entire dataset at once then leverage DDP and Dataloader.