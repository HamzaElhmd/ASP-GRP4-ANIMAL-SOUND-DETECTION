
import matplotlib.pyplot as plt

# Data from glob
data_counts = {
    "background": 1137,
    "cat": 2652,
    "cow": 756,
    "dog": 1896,
    "rooster": 720,
    "sheep": 666
}

# Create a bar chart
plt.figure(figsize=(10, 6))
plt.bar(data_counts.keys(), data_counts.values())
plt.title("Class Distribution in processed/ folder")
plt.xlabel("Class")
plt.ylabel("Number of files")
plt.savefig("processed_class_imbalance.png")
print("Plot saved to processed_class_imbalance.png")
