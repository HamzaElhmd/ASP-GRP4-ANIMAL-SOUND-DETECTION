
import matplotlib.pyplot as plt

# Data from glob
data_counts = {
    "background": 150,
    "cat": 240,
    "cow": 115,
    "dog": 240,
    "rooster": 80,
    "sheep": 83
}

# Create a bar chart
plt.figure(figsize=(10, 6))
plt.bar(data_counts.keys(), data_counts.values())
plt.title("Class Distribution in dataset/ folder")
plt.xlabel("Class")
plt.ylabel("Number of files")
plt.savefig("dataset_class_imbalance.png")
print("Plot saved to dataset_class_imbalance.png")
