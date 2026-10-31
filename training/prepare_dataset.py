import csv #let's python reads csv's
from pathlib import Path #to locate files location safely
import numpy as np
import json


dataset_dir = Path("../datasets/dataset_v0") #actual folder path
labels_file = dataset_dir / "labels.csv" # / means - appending labels.csv to folder path
image_dir = dataset_dir / "images"

#using dictreader to read csv contents
samples = [] #to store pairs
with open(labels_file , "r", newline="") as file:

    reader = csv.DictReader(file)

    for row in reader:
        image_id = row["image_id"]
        theta_E = float(row["theta_E"])
        #csv normally sees them as strings
        #but CNN needs theta_E as numbers so
        #we explicitly convert them and image_id
        #is just identifier so it can be string itself
        samples.append((image_id,theta_E))

#since the theta_E values are uniform so we shuffle them and split into sets
rng = np.random.default_rng(42)
rng.shuffle(samples)

#now split is 80% training , 10% validation , 10% testing

train_samples = samples[:8000]
validation_samples = samples[8000:9000]
test_samples = samples[9000:]

x_train = []
y_train = []

for image_id , theta_E in train_samples:
    image_path = image_dir / f"{image_id}.npy"
    image = np.load(image_path)

    x_train.append(image)
    y_train.append(theta_E)

#converting into numpy arrays so that standardisation can be done
x_train = np.array(x_train)
y_train = np.array(y_train)

mean = x_train.mean()
std = x_train.std()

metadata = {
    "dataset_version": "dataset_v0",
    "model_version": "olin_v0",
    "normalization": {
        "mean": float(mean),
        "std": float(std)
    }
}

metadata_file = dataset_dir / "metadata.json"

with open(metadata_file, "w") as file:
    json.dump(metadata, file, indent=4)

print(f"metadata saved to: {metadata_file}")

#Standardisation 
x_train = (x_train - mean) / std

#about to standardise the validation and testing sets 
# using the training-set mean and standard deviation

x_validation = []
y_validation = []

for image_id, theta_E in validation_samples:
    image_path = image_dir / f"{image_id}.npy"
    image = np.load(image_path)

    x_validation.append(image)
    y_validation.append(theta_E)

x_validation = np.array(x_validation)
y_validation = np.array(y_validation)


x_test = []
y_test = []

for image_id, theta_E in test_samples:
    image_path = image_dir / f"{image_id}.npy"
    image = np.load(image_path)

    x_test.append(image)
    y_test.append(theta_E)

x_test = np.array(x_test)
y_test = np.array(y_test)

x_validation = (x_validation - mean) / std #validation set standardisation
x_test = (x_test - mean) / std #testing  set standardisation

prepared_dir = Path("../datasets")

np.save(prepared_dir / "x_train_0.npy", x_train)
np.save(prepared_dir / "y_train_0.npy", y_train)

np.save(prepared_dir / "x_validation_0.npy", x_validation)
np.save(prepared_dir / "y_validation_0.npy", y_validation)

np.save(prepared_dir / "x_test_0.npy", x_test)
np.save(prepared_dir / "y_test_0.npy", y_test)

print("prepared arrays saved.")


        

