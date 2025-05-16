import os
import time
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Parameters
base_path = "out/spikes_outputs/sample_00_epoch_00_label_00"
time_window_sec = 4
update_interval_sec = 0.5

layer_dirs = sorted([d for d in os.listdir(base_path) if d.startswith("layer_")])
layer_count = len(layer_dirs)
layer_spikes = [{} for _ in range(layer_count)]  # One dict per layer: {neuron_idx: [times]}

fig, axes = plt.subplots(nrows=layer_count, sharex=True, figsize=(10, 2 * layer_count))
if layer_count == 1:
    axes = [axes]  # Ensure list format

fig.suptitle("Live Spike Raster (Last 4 Seconds)")

def parse_spike_file(filepath):
    with open(filepath, "r") as f:
        lines = f.readlines()
        if len(lines) < 3:
            return []

        timestamp_us = int(os.path.basename(filepath).split("_")[1].split(".")[0])
        timestamp_sec = timestamp_us / 1e6

        spikes = []
        for neuron_idx, line in enumerate(lines[2:]):
            if float(line.strip()) == 1.0:
                spikes.append((neuron_idx, timestamp_sec))
        return spikes

def update(_):
    global layer_spikes

    now_sec = time.time()
    min_time = now_sec - time_window_sec

    for layer_idx, layer_dir in enumerate(layer_dirs):
        layer_path = os.path.join(base_path, layer_dir)
        files = sorted(glob.glob(os.path.join(layer_path, "t_*.txt")), key=os.path.getmtime)

        for f in files[-30:]:  # Only check last few files
            new_spikes = parse_spike_file(f)
            for neuron_idx, spike_time in new_spikes:
                if neuron_idx not in layer_spikes[layer_idx]:
                    layer_spikes[layer_idx][neuron_idx] = []
                if spike_time not in layer_spikes[layer_idx][neuron_idx]:  # Prevent duplicate
                    layer_spikes[layer_idx][neuron_idx].append(spike_time)

        # Trim old spikes
        for neuron_idx in list(layer_spikes[layer_idx].keys()):
            layer_spikes[layer_idx][neuron_idx] = [
                t for t in layer_spikes[layer_idx][neuron_idx] if t >= min_time
            ]

        # Plotting
        ax = axes[layer_idx]
        ax.clear()
        ax.set_title(f"Layer {layer_idx}")
        ax.set_ylabel("Neuron")
        ax.set_xlim(min_time, now_sec)

        spike_times = []
        neuron_indices = []

        for neuron_idx, times in layer_spikes[layer_idx].items():
            spike_times.append(times)
            neuron_indices.append(neuron_idx)

        if spike_times:
            ax.eventplot(spike_times, lineoffsets=neuron_indices, colors='black')

    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    fig.subplots_adjust(top=0.93)

ani = FuncAnimation(fig, update, interval=int(update_interval_sec * 1000))
plt.show()
