from huggingface_hub import snapshot_download
import pandas as pd
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
import io
from PIL import Image
import random
import yaml
import os


snapshot_download(
    repo_id="kmewhort/tu-berlin-png",
    repo_type="dataset",
    local_dir=r"D:\TuBerlin"
)

exit()

# Load label decoder from README.md
def load_label_decoder(readme_path):
    with open(readme_path, "r", encoding="utf-8") as f:
        content = f.read()
    # Find the section with class_label names
    start = content.find("class_label:")
    if start == -1:
        raise ValueError("class_label section not found in README.md")
    start = content.find("names:", start)
    if start == -1:
        raise ValueError("names section not found in README.md")
    end = content.find("splits:", start)
    yaml_str = content[start:end]
    # Add a dummy root to parse as YAML
    yaml_str = "names:\n" + "\n".join(line for line in yaml_str.splitlines()[1:])
    names = yaml.safe_load(yaml_str)["names"]
    # Convert to int->str mapping
    return {int(k): v for k, v in names.items()}

# Find a parquet file to sample from
data_dir = r"D:\TuBerlin\data"
parquet_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".parquet")]
if not parquet_files:
    raise FileNotFoundError("No parquet files found in data directory.")

# Load a random parquet file and sample some rows
parquet_path = random.choice(parquet_files)
table = pq.read_table(parquet_path)
df = table.to_pandas()

# Sample N*N random examples for an 8x8 grid
N = 8
sampled = df.sample(N * N)

# Load label decoder
label_decoder = load_label_decoder(r"D:\TuBerlin\README.md")

# Plot the images with labels in an 8x8 matrix
fig, axes = plt.subplots(N, N, figsize=(16, 16))
for idx, (img_dict, label) in enumerate(zip(sampled["image"], sampled["label"])):
    row = idx // N
    col = idx % N
    ax = axes[row, col]
    img_bytes = img_dict["bytes"]
    img = Image.open(io.BytesIO(img_bytes))
    img = img.resize((224, 224))
    ax.imshow(img, cmap="gray")
    ax.axis("off")
    ax.set_title(label_decoder.get(label, str(label)), fontsize=8)
plt.tight_layout()
plt.show()

