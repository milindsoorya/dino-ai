
import keras
from keras import layers
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt

(x_train, _), (x_test, _) = mnist.load_data()

# Importing data
num_data = 1000
frac_train = 0.7
frac_test = 0.3
x_train = x_train[0:int(frac_train*num_data)]
x_test = x_test[0:int(frac_test*num_data)]
dimension = x_train.shape[1]

# Pre-processing data
norm_factor = 255.
x_train = x_train.astype('float32')/norm_factor
x_test = x_test.astype('float32')/norm_factor
x_train = np.reshape(x_train, (len(x_train), dimension, dimension, 1))
x_test = np.reshape(x_test, (len(x_test), dimension, dimension, 1))

# Add noise to the images
noise_factor = 0.5
x_train_noisy = x_train + noise_factor * \
    np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
x_test_noisy = x_test + noise_factor * \
    np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)

x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)


# Visualise the noisy images
n = 3
for i in range(n):
    fig, axes = plt.subplots(1, 2)
    fig.set_size_inches(5, 5)
    axes[0].set_title('True image')
    im0 = axes[0].imshow(x_test[i].reshape(dimension, dimension), cmap='Reds')
    axes[1].set_title('Noisy image')
    im1 = axes[1].imshow(x_test_noisy[i].reshape(
        dimension, dimension), cmap='Reds')


# Building the Autoencoder
input_img = keras.Input(shape=(dimension, dimension, 1))

# Encoder
x = layers.Conv2D(filters=32, kernel_size=(
    3, 3), activation='relu', padding='same')(input_img)
x = layers.MaxPooling2D(pool_size=(2, 2), padding='same')(x)
x = layers.Conv2D(filters=32, kernel_size=(
    3, 3), activation='relu', padding='same')(x)
encoded = layers.MaxPooling2D(pool_size=(2, 2), padding='same')(x)

# Decoder
x = layers.Conv2D(filters=32, kernel_size=(
    3, 3), activation='relu', padding='same')(encoded)
x = layers.UpSampling2D(size=(2, 2))(x)
x = layers.Conv2D(filters=32, kernel_size=(
    3, 3), activation='relu', padding='same')(x)
x = layers.UpSampling2D(size=(2, 2))(x)
decoded = layers.Conv2D(filters=1, kernel_size=(
    3, 3), activation='sigmoid', padding='same')(x)

#  Autoencoder Model
autoencoder = keras.Model(input_img, decoded)

# Model Summary
autoencoder.summary()

# Compiling the model
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# Fitting
validation_split = 0.8
history = autoencoder.fit(x_train_noisy, x_train, epochs=40,
                          batch_size=20, shuffle=True, validation_split=validation_split)

train_loss = history.history['loss']
train_val_loss = history.history['val_loss']
epochs = range(1, len(train_loss) + 1)

# Plotting the loss
plt.figure(dpi=100)
plt.plot(epochs, train_loss, label='Loss')
plt.plot(epochs, train_val_loss, 'o', label='Val loss')
plt.title('Training and validation metrics')
plt.legend()
plt.savefig('history.png')

#  Prediction
all_denoised_images = autoencoder.predict(x_test_noisy)

# Evaluating the model
test_loss = autoencoder.evaluate(x_test_noisy, x_test, batch_size=20)
test_loss

# Visualising the results
n = 3
for i in range(n):
    fig, axes = plt.subplots(1, 3)
    fig.set_size_inches(8, 2)
    axes[0].set_title('Noisy image')
    im0 = axes[0].imshow(x_test_noisy[i].reshape(
        dimension, dimension), cmap='Reds')
    axes[1].set_title('Target image')
    im1 = axes[1].imshow(x_test[i].reshape(dimension, dimension), cmap='Reds')
    axes[2].set_title('Denoised image')
    im2 = axes[2].imshow(all_denoised_images[i].reshape(
        dimension, dimension), cmap='Reds')
    plt.savefig(f'comparison-{i}.png')
