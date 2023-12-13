import pandas as pd
import matplotlib.pyplot as plt

# Replace this with the path to your CSV file
csv_file = "/home/denhofja/gpu_stats_.csv"

# Read the CSV file
data = pd.read_csv(csv_file)

# Convert 'Timestamp' to datetime for better plotting
data['Timestamp'] = pd.to_datetime(data['Timestamp'], format="ISO8601").apply(lambda x: x.timestamp())

# Plotting
plt.figure(figsize=(10, 6))

# Plot each metric in a subplot
plt.subplot(2, 2, 1)
plt.plot(data['Timestamp'], data['Free Memory (MB)'])
plt.title('Free Memory (MB) over Time')
plt.xticks(rotation=45)

plt.subplot(2, 2, 2)
plt.plot(data['Timestamp'], data['GPU Utilization (%)'])
plt.title('GPU Utilization (%) over Time')
plt.xticks(rotation=45)

plt.subplot(2, 2, 3)
plt.plot(data['Timestamp'], data['Power Draw (W)'])
plt.title('Power Draw (W) over Time')
plt.xticks(rotation=45)

# Adjust layout
plt.tight_layout()

# Save the plot to a file
plt.savefig("/home/denhofja/system_stats_plot.png", dpi=300, bbox_inches='tight')
