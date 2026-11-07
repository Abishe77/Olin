from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt

folder = Path("/mnt/d/Olin/real_data/HE0435-1223")

original = Image.open(folder / "original_HE0435H.gif").convert("L")
cleaned = Image.open(folder / "cleaned_HE0435Hcc.gif").convert("L")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].imshow(original, cmap="gray")
axes[0].set_title("Original HE0435 H-band")

axes[1].imshow(cleaned, cmap="gray")
axes[1].set_title("Cleaned HE0435 H-band")

for ax in axes:
    ax.axis("off")

plt.tight_layout()
plt.savefig(folder / "comparison.png", dpi=150)
plt.show()

for name, img in [("Original", original), ("Cleaned", cleaned)]:
    print(
        name,
        "Min:", min(img.getdata()),
        "Max:", max(img.getdata()),
        "Mean:", sum(img.getdata()) / (img.width * img.height)
    )