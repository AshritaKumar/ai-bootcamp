import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix
)
import numpy as np

# True and predicted labels
y_true = [1, 0, 1, 1, 0, 1]
y_pred = [1, 0, 1, 0, 0, 1]

# Metrics
acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred)
rec = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print("Accuracy:", acc)
print("Precision:", prec)
print("Recall:", rec)
print("F1 Score:", f1)

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
labels = ["Negative", "Positive"]

# --- Plot Confusion Matrix ---
fig, ax = plt.subplots(1, 2, figsize=(10, 4))

im = ax[0].imshow(cm, cmap="Blues")
ax[0].set_xticks(np.arange(len(labels)))
ax[0].set_yticks(np.arange(len(labels)))
ax[0].set_xticklabels(labels)
ax[0].set_yticklabels(labels)
ax[0].set_xlabel("Predicted Label")
ax[0].set_ylabel("True Label")
ax[0].set_title("Confusion Matrix")

# Annotate each cell
for i in range(len(labels)):
    for j in range(len(labels)):
        ax[0].text(j, i, cm[i, j], ha="center", va="center", color="black")

plt.colorbar(im, ax=ax[0])

# --- Plot Metrics Bar Chart ---
metrics_names = ["Accuracy", "Precision", "Recall", "F1-Score"]
metrics_values = [acc, prec, rec, f1]

ax[1].bar(metrics_names, metrics_values, color=["#4CAF50", "#2196F3", "#FFC107", "#E91E63"])
ax[1].set_ylim(0, 1)
ax[1].set_title("Evaluation Metrics")
for i, v in enumerate(metrics_values):
    ax[1].text(i, v + 0.02, f"{v:.2f}", ha="center")

plt.tight_layout()
plt.show()
