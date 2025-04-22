import os
import numpy as np
import matplotlib.pyplot as plt

def plot_spike_raster(layer_path, layer_num, sample, epoch):
    spike_data = {}

    for filename in sorted(os.listdir(layer_path)):
        if filename.startswith("t_") and filename.endswith(".txt"):
            t = int(filename.split("_")[1].split(".")[0])  
            filepath = os.path.join(layer_path, filename)

            with open(filepath, "r") as file:
                lines = file.readlines()
                spike_values = [float(line.strip()) for line in lines[2:]]

                for neuron_idx, spike in enumerate(spike_values):
                    if spike == 1.0:
                        if neuron_idx not in spike_data:
                            spike_data[neuron_idx] = []
                        spike_data[neuron_idx].append(t)

    if not spike_data:
        print(f"No spikes detected in {layer_path}")
        return

    spike_times = []
    neuron_indices = []
    for neuron_idx, times in spike_data.items():
        spike_times.append(times)  # List of spike times for this neuron
        neuron_indices.append(neuron_idx)  # Corresponding neuron index

    # Create raster plot
    plt.figure(figsize=(10, 6))
    plt.eventplot(spike_times, lineoffsets=neuron_indices, colors='black')
    plt.xlabel("Time (t)")
    plt.ylabel("Neuron Index")
    plt.title(f"Spike Raster Plot - Layer {layer_num} (Sample {sample}, Epoch {epoch})")
    plt.savefig(f"{layer_path}/spike_raster_layer_{layer_num}.png")
    plt.close()

def visualize_spikes(sample, epoch):
    base_path = f"out/spikes_outputs/sample_{sample:02d}_epoch_{epoch:02d}"
    if not os.path.exists(base_path):
        print(f"Path {base_path} does not exist.")
        return

    for layer_dir in sorted(os.listdir(base_path)):
        if layer_dir.startswith("layer_"):
            layer_num = int(layer_dir.split("_")[1])  
            layer_path = os.path.join(base_path, layer_dir)
            print(f"Processing {layer_path}")
            plot_spike_raster(layer_path, layer_num, sample, epoch)

visualize_spikes(sample=0, epoch=1)