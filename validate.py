import os

def process_files(directory):
    for filename in os.listdir(directory):
        if filename.startswith("inputs_epoch_") and filename.endswith(".txt"):
            filepath = os.path.join(directory, filename)
            with open(filepath, 'r') as file:
                lines = file.readlines()

            # Skip the header (first 2 lines)
            values = []
            for line in lines[2:]:
                try:
                    val = float(line.strip())
                    if val > 0.0:
                        values.append(val)
                except ValueError:
                    continue

            if values:
                print({
                    "filename": filename,
                    "non_zero_values": values
                })

process_files("C:/Users/karol/Desktop/projects/snn-generator/out/inputs/Conv2D_0")