
import pandas as pd
import matplotlib.pyplot as plt

# Read the farmyard.csv file
df = pd.read_csv('dataset/farmyard.csv')

# Calculate the total duration for each class
duration_per_class = df.groupby('label')['duration'].sum()

# Create a bar chart
plt.figure(figsize=(10, 6))
duration_per_class.plot(kind='bar')
plt.title('Total Duration per Class')
plt.xlabel('Class')
plt.ylabel('Total Duration (seconds)')
plt.savefig('audit_duration_plot.png')

print("Audit plot saved to audit_duration_plot.png")
