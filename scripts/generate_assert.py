"""
Generate all visual assets for the MNIST Linearity Project, using pre-computed/simulated data.
Saves figures under ./assets/ for use in README.md
"""
import sys
from pathlib import Path
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.models import LinearClassifier, MLP
from src.utils import accuracy_from_logits # This import is not used, but kept for reference

os.makedirs("assets", exist_ok=True)

# ============================================================
# 1️⃣ Accuracy comparison bar chart
# ============================================================
results = {
    "Linear (MSE)": 92.57,
    "Linear (CE)": 92.15,
    "MLP (ReLU)": 97.45,
    "MLP + Dropout": 97.28,
    "MLP + BatchNorm": 97.66,
}

plt.figure(figsize=(8, 4))
sns.barplot(x=list(results.keys()), y=list(results.values()), palette="viridis")
plt.ylabel("Accuracy (%)")
plt.title("Model Performance Comparison")
plt.xticks(rotation=25, ha="right")
plt.tight_layout()
plt.savefig("assets/comparison_chart.png", dpi=300)
plt.close()
print("✅ Generated accuracy comparison chart.")

# ============================================================
# 2️⃣ Example training curves (simulated)
# ============================================================
epochs = np.arange(1, 6)
loss_linear = [0.03, 0.02, 0.015, 0.013, 0.012]
loss_mlp = [0.37, 0.16, 0.11, 0.08, 0.06]
acc_linear = [84, 90, 91, 92, 92.5]
acc_mlp = [90, 95, 96.5, 97.3, 97.5]

fig, ax1 = plt.subplots(figsize=(6, 4))
ax2 = ax1.twinx()
ax1.plot(epochs, loss_linear, "o--", label="Linear Loss", color="tab:red")
ax1.plot(epochs, loss_mlp, "s--", label="MLP Loss", color="tab:orange")
ax2.plot(epochs, acc_linear, "o-", label="Linear Acc", color="tab:blue")
ax2.plot(epochs, acc_mlp, "s-", label="MLP Acc", color="tab:green")

ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax2.set_ylabel("Accuracy (%)")
ax1.set_title("Training Dynamics: Linear vs MLP")
ax1.legend(loc="upper left")
ax2.legend(loc="lower right")

plt.tight_layout()
plt.savefig("assets/linear_curve.png", dpi=300)
plt.close()
print("✅ Generated training curves plot.")

# ============================================================
# 3️⃣ Linear weight visualization
# ============================================================
model_path = "mnist_linear_crossentropy_best.pt"
if os.path.exists(model_path):
    model = LinearClassifier()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    W = model.fc.weight.detach().cpu().numpy()

    fig, axes = plt.subplots(2, 5, figsize=(10, 4))
    for i, ax in enumerate(axes.flat):
        ax.imshow(W[i].reshape(28, 28), cmap="coolwarm")
        ax.set_title(f"Class {i}")
        ax.axis("off")
    plt.suptitle("Linear Model Weight Templates")
    plt.tight_layout()
    plt.savefig("assets/weights_linear.png", dpi=300)
    plt.close()
    print("✅ Generated linear model weight visualization.")
else:
    print("[Warning] No 'mnist_linear_crossentropy_best.pt' found, skipping weight visualization.")

print("✅ All figures generated in ./assets/")

import matplotlib.pyplot as plt

# 模拟训练结果（可替换成你真实日志数据）
epochs = [1, 2, 3, 4, 5]
train_loss = [0.37, 0.16, 0.11, 0.085, 0.065]
val_acc = [93.8, 95.5, 96.3, 96.8, 97.1]

fig, ax1 = plt.subplots(figsize=(6, 4))

ax1.plot(epochs, train_loss, 'o-', color='tomato', label='Train Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss', color='tomato')
ax1.tick_params(axis='y', labelcolor='tomato')

ax2 = ax1.twinx()
ax2.plot(epochs, val_acc, 's-', color='royalblue', label='Val Accuracy')
ax2.set_ylabel('Accuracy (%)', color='royalblue')
ax2.tick_params(axis='y', labelcolor='royalblue')

plt.title('Training Dynamics: MLP + ReLU')
fig.tight_layout()
plt.savefig('assets/mlp_curve.png', dpi=150)
plt.close()
print("✅ Saved assets/mlp_curve.png")
