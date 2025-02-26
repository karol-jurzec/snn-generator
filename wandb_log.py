import numpy as np
import wandb
import glob

wandb.init(project="snn-generator", entity="jurzeckarol-agh-ust")

# Read all weight files
for filename in sorted(glob.glob("out/weights/weights_epoch_*.txt")):
    with open(filename, "r") as f:
        lines = f.readlines()

    epoch = int(lines[0].split()[1])  # Extract epoch number
    weights = [float(line) for line in lines[2:] if not line.startswith("Layer")]


    print(weights)

    # Log histogram of weights
    # wandb.log({"weights_histogram": wandb.Histogram(np.array(weights)), "epoch": epoch})
    