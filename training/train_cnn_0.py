#Loading the prepared datasets
import numpy as np
from pathlib import Path
import tensorflow as tf #for cnn
tf.keras.utils.set_random_seed(42) #to reproduce the same results


datasets_dir = Path("../datasets")

x_train = np.load(datasets_dir / "x_train_0.npy")
y_train = np.load(datasets_dir / "y_train_0.npy")


x_validation = np.load(datasets_dir/ "x_validation_0.npy")
y_validation = np.load(datasets_dir / "y_validation_0.npy")

x_test = np.load(datasets_dir / "x_test_0.npy")
y_test = np.load(datasets_dir / "y_test_0.npy")


#now the shapes are basically(no of sample , no of array rows , no of array cols)
#but CNN needs four things and the last one is channel
#normallt RGB images have 3 , but since this is
#scinetific grayscale image we'll keep it 1
#so the shape becomes(s,m,n,1)

x_train = x_train[...,np.newaxis] 
x_validation = x_validation[...,np.newaxis]
x_test = x_test[...,np.newaxis]
# ... means keep the size same , newaxis just one dimension extra

model = tf.keras.Sequential([

    tf.keras.Input(shape=(160,160,1)),

    tf.keras.layers.Conv2D(16,(3,3) , activation = "relu"),
    tf.keras.layers.MaxPooling2D((2,2)),
    #16 different filters of size 3x3 , so op has 16 channels

    tf.keras.layers.Conv2D(32,(3,3) , activation = "relu"),
    tf.keras.layers.MaxPooling2D((2,2)),
    #takes those 16 channels or feature maps and learns 32 new filters

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(1)
])

model.summary()
model.compile(
    optimizer = "adam", #to adjust weights efficiently during training
    loss = "mse", #squared error
    metrics = ["mae"] #absolute error
)

history = model.fit(
    x_train,
    y_train,
    validation_data = (x_validation,y_validation),
    epochs =10,
    batch_size = 32
)

test_loss , test_mae = model.evaluate(
    x_test,
    y_test,
    verbose = 1
)

models_dir = Path("models")
models_dir.mkdir(exist_ok =True)
model.save(models_dir / "cnn_0.keras")

print("CNN_0 model saved") #just for my indication

print("Test MSE:", test_loss)
print("Test MAE:", test_mae)

predictions = model.predict(x_test)

for i in range(10):
    print(
        f"True: {y_test[i]:.4f} | "
        f"Predicted: {predictions[i, 0]:.4f} | "
        f"Error: {predictions[i, 0] - y_test[i]:.4f}"
    )
    
