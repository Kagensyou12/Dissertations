import matplotlib.pyplot as plt
import numpy as np

# Combine epochs of phase 1 and phase 2
epochs = np.arange(1, 18)  # 7 + 10 = 17 epochs total

# Combine metrics from phase 1 and phase 2
acc = [0.5033, 0.5047, 0.5046, 0.5094, 0.5090, 0.5143, 0.5213] + [0.5168, 0.5675, 0.6233, 0.6323, 0.6387, 0.6455, 0.6472, 0.6515, 0.6627, 0.6729]
val_acc = [0.5, 0.5274, 0.5, 0.5031, 0.5, 0.6158, 0.6107] + [0.5, 0.5002, 0.5029, 0.5304, 0.5893, 0.5583, 0.7316, 0.6180, 0.7395, 0.7344]

loss = [0.9520, 0.8544, 0.7394, 0.6984, 0.6976, 0.6954, 0.6945] + [0.7321, 0.7309, 0.7197, 0.7062, 0.7109, 0.7364, 0.7518, 0.7531, 0.7101, 0.6559]
val_loss = [0.6887, 0.6885, 0.6916, 0.6921, 0.6920, 0.6904, 0.6895] + [0.6923, 1.3644, 0.9666, 0.7355, 0.6372, 0.8251, 0.5211, 0.6523, 0.5182, 0.5229]

auc = [0.5032, 0.5046, 0.5041, 0.5107, 0.5102, 0.5180, 0.5250] + [0.5221, 0.5894, 0.6635, 0.6779, 0.6943, 0.7019, 0.7068, 0.7076, 0.7249, 0.7372]
val_auc = [0.7369, 0.7212, 0.6689, 0.5656, 0.6711, 0.6899, 0.7134] + [0.7638, 0.7704, 0.7806, 0.8122, 0.8069, 0.8082, 0.8434, 0.8255, 0.8455, 0.8438]

plt.figure(figsize=(18, 5))

plt.subplot(1, 3, 1)
plt.plot(epochs, acc, label='Train Accuracy')
plt.plot(epochs, val_acc, label='Val Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.grid(True)

plt.subplot(1, 3, 2)
plt.plot(epochs, loss, label='Train Loss')
plt.plot(epochs, val_loss, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)

plt.subplot(1, 3, 3)
plt.plot(epochs, auc, label='Train AUC')
plt.plot(epochs, val_auc, label='Val AUC')
plt.xlabel('Epoch')
plt.ylabel('AUC')
plt.title('Training and Validation AUC')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
