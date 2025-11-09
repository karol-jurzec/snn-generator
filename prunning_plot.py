import pandas as pd
import matplotlib.pyplot as plt

# Wczytaj dane z pliku CSV
df = pd.read_csv("stmnist_pruning_threshold_study.csv")

# Accuracy vs Threshold
plt.figure(figsize=(7, 5))
plt.plot(df["threshold"], df["accuracy_after"], marker="o", label="Accuracy after pruning")
plt.xlabel("Threshold")
plt.ylabel("Accuracy [%]")
plt.title("Impact of ST-MNIST prunning threshold on accuracy")
plt.legend()
plt.savefig("accuracy_vs_threshold.png", dpi=300)
plt.show()

# Speedup vs Threshold
plt.figure(figsize=(7, 5))
plt.plot(df["threshold"], df["speedup"], marker="o", color="orange", label="Speedup")
plt.xlabel("Threshold")
plt.ylabel("Speedup (x)")
plt.title("Impact of ST-MNIST prunning threshold on speedup")
plt.legend()
plt.savefig("speedup_vs_threshold.png", dpi=300)
plt.show()

# Pruning % vs Threshold
plt.figure(figsize=(7, 5))
plt.plot(df["threshold"], df["pruning_percentage"], marker="o", color="green", label="Pruning %")
plt.xlabel("Threshold")
plt.ylabel("Pruning Percentage [%]")
plt.title("Impact of ST-MNIST prunning threshold on prunning percentage")
plt.legend()
plt.savefig("pruning_percentage_vs_threshold.png", dpi=300)
plt.show()
