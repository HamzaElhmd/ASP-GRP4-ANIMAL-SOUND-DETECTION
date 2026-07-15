
import matplotlib.pyplot as plt
import numpy as np

# Data from the model results
models = ['HMM', 'CNN', 'YAMNet (calibrated)']
classes = ['cat', 'cow', 'dog', 'rooster', 'sheep']

# F1 scores for each model and class
hmm_f1 = [0.04, 0.69, 0.68, 0.71, 0.65]
cnn_f1 = [0.312, 0.100, 0.503, 0.325, 0.613]
yamnet_f1 = [0.732, 0.709, 0.738, 0.553, 0.829]

# Calculate the average F1 score for each class
avg_f1 = np.mean([hmm_f1, cnn_f1, yamnet_f1], axis=0)

# Create a list of tuples (class, avg_f1, hmm_f1, cnn_f1, yamnet_f1)
class_data = list(zip(classes, avg_f1, hmm_f1, cnn_f1, yamnet_f1))

# Sort the list by the average F1 score in descending order
class_data.sort(key=lambda x: x[1], reverse=True)

# Unpack the sorted data
sorted_classes, _, sorted_hmm_f1, sorted_cnn_f1, sorted_yamnet_f1 = zip(*class_data)

color_map = {'HMM': 'cadetblue', 'CNN': 'darksalmon', 'YAMNet (calibrated)': 'darkseagreen'}

x = np.arange(len(sorted_classes))  # the label locations
width = 0.2  # the width of the bars

fig, ax = plt.subplots(figsize=(12, 8))

for i, class_name in enumerate(sorted_classes):
    class_scores = {
        'HMM': sorted_hmm_f1[i],
        'CNN': sorted_cnn_f1[i],
        'YAMNet (calibrated)': sorted_yamnet_f1[i]
    }

    sorted_models = sorted(class_scores.items(), key=lambda item: item[1], reverse=True)

    for j, (model_name, score) in enumerate(sorted_models):
        ax.bar(x[i] + (j - 1) * width, score, width, label=model_name if i == 0 else "", color=color_map[model_name])

# Add some text for labels, title and axes ticks
ax.set_ylabel('Event-based F1 Score')
ax.set_title('Model Comparison by Class (Sorted by F1 Score for Each Class)')
ax.set_xticks(x)
ax.set_xticklabels(sorted_classes)

# Create custom legend
from matplotlib.lines import Line2D
legend_elements = [Line2D([0], [0], color=color_map[m], lw=4, label=m) for m in models]
ax.legend(handles=legend_elements)


fig.tight_layout()

plt.savefig('model_comparison_sorted_by_class_muted.png')
print("Generated model_comparison_sorted_by_class_muted.png")
