
from PIL import Image
import numpy as np

folder = "/mnt/d/Olin/real_data/HE0435-1223"

files = [
    "original_HE0435H.gif",
    "cleaned_HE0435Hcc.gif"
]

for filename in files:
    image = Image.open(f"{folder}/{filename}").convert("L")
    pixels = np.array(image)

    # Find bright pixels, keeping detected positions separated.
    candidates = np.argsort(pixels.ravel())[::-1]

    peaks = []

    for index in candidates:
        y, x = np.unravel_index(index, pixels.shape)

        if all(
            (x - px)**2 + (y - py)**2 >= 15**2
            for px, py, _ in peaks
        ):
            peaks.append((x, y, int(pixels[y, x])))

        if len(peaks) == 8:
            break

    print(f"\n{filename}")
    print("Image centre:", (pixels.shape[1] // 2,
                            pixels.shape[0] // 2))
    print("Brightest separated locations:")

    for x, y, intensity in peaks:
        print(f"x={x:3}, y={y:3}, intensity={intensity}")
