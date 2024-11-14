import pandas as pd
import numpy as np

# Create a synthetic dataset
np.random.seed(42)

# Define the number of rows (data points)
num_rows = 1000

# Generate random values for each feature
packet_size = np.random.randint(40, 1500, num_rows)  # Packet size (between 40 and 1500 bytes)
protocol = np.random.choice(['TCP', 'UDP', 'ICMP'], num_rows)  # Random protocols (TCP, UDP, ICMP)
duration = np.random.randint(1, 600, num_rows)  # Duration of the packet flow in seconds
source_ip = np.random.choice([f"192.168.1.{i}" for i in range(1, 255)], num_rows)  # Random source IP addresses
destination_ip = np.random.choice([f"10.0.0.{i}" for i in range(1, 255)], num_rows)  # Random destination IP addresses
label = np.random.choice([0, 1], num_rows)  # 0 = Benign, 1 = Malicious

# Combine all the features into a pandas DataFrame
data = pd.DataFrame({
    'PacketSize': packet_size,
    'Protocol': protocol,
    'Duration': duration,
    'SourceIP': source_ip,
    'DestinationIP': destination_ip,
    'Label': label
})

# Save the dataset to a CSV file
file_path = 'network_traffic_data.csv'
data.to_csv(file_path, index=False)

print(f"Synthetic dataset saved as {file_path}")
