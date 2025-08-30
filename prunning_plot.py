import pandas as pd
import matplotlib.pyplot as plt

# Wczytaj dane z pliku CSV
df = pd.read_csv("stmnist_pruning_threshold_study.csv")

# Tworzenie wykresów
plt.figure(figsize=(15, 5))

# Accuracy vs Threshold
plt.subplot(1, 3, 1)
plt.plot(df["threshold"], df["accuracy_after"], marker="o", label="Accuracy after pruning")
plt.xlabel("Threshold")
plt.ylabel("Accuracy [%]")
plt.title("Accuracy vs Threshold")
plt.grid(True)
plt.legend()

# Speedup vs Threshold
plt.subplot(1, 3, 2)
plt.plot(df["threshold"], df["speedup"], marker="o", color="orange", label="Speedup")
plt.xlabel("Threshold")
plt.ylabel("Speedup (x)")
plt.title("Speedup vs Threshold")
plt.grid(True)
plt.legend()

# Pruning % vs Threshold
plt.subplot(1, 3, 3)
plt.plot(df["threshold"], df["pruning_percentage"], marker="o", color="green", label="Pruning %")
plt.xlabel("Threshold")
plt.ylabel("Pruning Percentage [%]")
plt.title("Pruning Percentage vs Threshold")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig("pruning_analysis_clean.png", dpi=300)  # zapis do pliku PNG
plt.show()