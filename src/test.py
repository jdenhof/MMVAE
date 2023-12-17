# import pandas as pd
# import matplotlib.pyplot as plt

# # Replace this with the path to your CSV file
# csv_file = "/home/denhofja/gpu_stats_.csv"

# # Read the CSV file
# data = pd.read_csv(csv_file)

# # Convert 'Timestamp' to datetime for better plotting
# data['Timestamp'] = pd.to_datetime(data['Timestamp'], format="ISO8601").apply(lambda x: x.timestamp())

# # Plotting
# plt.figure(figsize=(10, 6))

# # Plot each metric in a subplot
# plt.subplot(2, 2, 1)
# plt.plot(data['Timestamp'], data['Free Memory (MB)'])
# plt.title('Free Memory (MB) over Time')
# plt.xticks(rotation=45)

# plt.subplot(2, 2, 2)
# plt.plot(data['Timestamp'], data['GPU Utilization (%)'])
# plt.title('GPU Utilization (%) over Time')
# plt.xticks(rotation=45)

# plt.subplot(2, 2, 3)
# plt.plot(data['Timestamp'], data['Power Draw (W)'])
# plt.title('Power Draw (W) over Time')
# plt.xticks(rotation=45)

# # Adjust layout
# plt.tight_layout()

# # Save the plot to a file
# plt.savefig("/home/denhofja/system_stats_plot.png", dpi=300, bbox_inches='tight')

import torch
import multiprocessing as mp

def worker(proc_num, queue):
    # Worker process: Processes a tensor and sends it to the main process
    for _ in range(5):  # Just for demonstration, let's do this 5 times
        tensor = torch.randn(10)  # Create a random tensor
        queue.put(tensor)  # Send tensor to main process
        tensor = queue.get()  # Get the empty buffer (tensor) back

    print(f"Process {proc_num} completed.")

def main():
    # Main process
    mp.set_start_method('spawn')  # 'spawn' is safer when working with CUDA

    queue = mp.Queue()
    processes = []

    # Create and start worker processes
    for i in range(4):  # Let's say we have 4 worker processes
        p = mp.Process(target=worker, args=(i, queue))
        p.start()
        processes.append(p)

    # Main process retrieves tensors from the queue
    for _ in range(20):  # We expect 20 tensors in total (4 workers * 5 tensors each)
        tensor = queue.get()
        print(f"Received tensor: {tensor}")
        queue.put(torch.empty(0))  # Send back an empty tensor (buffer)

    # Wait for all processes to complete
    for p in processes:
        p.join()

if __name__ == "__main__":
    main()
