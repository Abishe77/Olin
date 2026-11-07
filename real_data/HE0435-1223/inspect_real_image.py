from PIL import Image
import numpy as np

folder = "/mnt/d/Olin/real_data/HE0435-1223"

files = [
    "original_HE0435H.gif",
    "cleaned_HE0435Hcc.gif"
]

for filename in files:
    image = Image.open(f"{folder}/{filename}")

    rgb = np.array(image.convert("RGB"))
    gray = np.array(image.convert("L"))

    print(f"\n{filename}")
    print("Dimensions:", gray.shape)
    print("RGB range:", rgb.min(), "to", rgb.max())
    print("Grayscale range:", gray.min(), "to", gray.max())
    print("Mean intensity:", gray.mean())
    print("Standard deviation:", gray.std())
    print("Unique grayscale values:", len(np.unique(gray)))