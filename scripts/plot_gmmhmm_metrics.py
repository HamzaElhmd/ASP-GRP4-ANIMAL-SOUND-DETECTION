
import json
import matplotlib.pyplot as plt
import numpy as np

with open('artifacts/final_diagnostics.json') as f:
    data = json.load(f)

event_based_metrics = data['event_based']
class_labels = list(event_based_metrics.keys())
precision = [event_based_metrics[label]['precision'] for label in class_labels]
recall = [event_based_metrics[label]['recall'] for label in class_labels]
f1 = [event_based_metrics[label]['f1'] for label in class_labels]

x = np.arange(len(class_labels))
width = 0.25

fig, ax = plt.subplots(figsize=(12, 7))
rects1 = ax.bar(x - width, precision, width, label='Precision')
rects2 = ax.bar(x, recall, width, label='Recall')
rects3 = ax.bar(x + width, f1, width, label='F1-score')

ax.set_ylabel('Scores')
ax.set_title('Event-based Metrics for GMM-HMM')
ax.set_xticks(x)
ax.set_xticklabels(class_labels)
ax.legend()

fig.tight_layout()
plt.savefig('gmmhmm_event_metrics.png')
print("Plot saved to gmmhmm_event_metrics.png")
