import numpy as np
import plotly.graph_objects as go
import glob

# Set epsilon threshold for near-zero filtering
epsilon = 1e-5

# Collect gradient files
gradient_files = sorted(glob.glob("out/weight_grads/gradients_epoch_*_sample_*.txt"))

# Data structure to store gradients per layer
layer_gradients = {}

# Read gradient files
for filename in gradient_files:
    with open(filename, "r") as f:
        lines = f.readlines()

    lines = lines[1:]  # Removes the first element
    sample_id = int(filename.split("_sample_")[-1].split(".txt")[0])  # Extract sample number
    current_layer = None

    for line in lines:
        line = line.strip()
        if line.startswith("Layer"):
            current_layer = int(line.split()[1])
            if current_layer not in layer_gradients:
                layer_gradients[current_layer] = []
        elif line:
            grad_value = float(line)
            if abs(grad_value) > epsilon:  # Ignore near-zero gradients
                layer_gradients[current_layer].append((sample_id, grad_value))

# Create interactive 3D scatter plots per layer
for layer, data in layer_gradients.items():
    data = np.array(data)  # Convert list to NumPy array
    sample_ids = data[:, 0]
    gradients = data[:, 1]

    if len(gradients) == 0:
        print(f"Skipping Layer {layer} (only near-zero gradients)")
        continue

    print(f"Layer {layer} - min gradient val: {min(gradients)}")
    print(f"Layer {layer} - max gradient val: {max(gradients)}")

    # Define histogram bins
    num_bins = 256
    hist_range = (min(gradients), max(gradients))
    hist, bins = np.histogram(gradients, bins=num_bins, range=hist_range)

    xpos = np.array([sample_ids[i] for i in range(len(gradients))])

    # Map gradient values to bin midpoints
    bin_indices = np.digitize(gradients, bins) - 1  # Get bin index
    bin_indices = np.clip(bin_indices, 0, num_bins - 1)  # Keep within valid range

    # Calculate bin midpoints
    bin_midpoints = (bins[:-1] + bins[1:]) / 2  
    ypos = bin_midpoints[bin_indices]  # Replace bin index with actual gradient value midpoint

    dz = hist[bin_indices]  # Frequency of each bin

    # Create a 3D scatter plot
    fig = go.Figure(data=[go.Scatter3d(
        x=xpos,
        y=ypos,
        z=dz,
        mode='markers',
        marker=dict(size=5, color=dz, colorscale='Viridis', opacity=0.8)
    )])

    # Labels
    fig.update_layout(
        title=f"Gradient Histogram for Layer {layer}",
        scene=dict(
            xaxis_title="Sample ID",
            yaxis_title="Gradient Value (Midpoint of Bin)",
            zaxis_title="Frequency"
        )
    )

    # Save as an interactive HTML file
    fig.write_html(f"gradient_histogram_layer_{layer}.html")

print("Saved all histograms as interactive HTML files.")
