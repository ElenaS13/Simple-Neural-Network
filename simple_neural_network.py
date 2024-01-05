import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler


def normalize_data(data):
  scaler = MinMaxScaler()
  normalized_data = scaler.fit_transform(data)
  return normalized_data, scaler


# Original data
house_sizes = np.array([[1400.0], [1600.0], [1800.0], [2000.0]],
                       dtype=np.float32)
actual_prices = np.array([[200000], [230000], [250000], [280000]],
                         dtype=np.float32)

# Normalize the training data
house_sizes_normalized, house_sizes_scaler = normalize_data(house_sizes)
actual_prices_normalized, actual_prices_scaler = normalize_data(actual_prices)

# Define a simple neural network with one neuron
model = tf.keras.Sequential(
    [tf.keras.layers.Dense(units=1, activation='linear', input_shape=(1, ))])

# Compile the model
model.compile(optimizer='sgd', loss='mean_squared_error')

# Train the model
model.fit(house_sizes_normalized,
          actual_prices_normalized,
          epochs=1000,
          verbose=0)

# Make predictions
predictions = model.predict(house_sizes_normalized)

# Plot the results
plt.scatter(house_sizes_normalized,
            actual_prices_normalized,
            label='Actual Data')
plt.plot(house_sizes_normalized, predictions, color='red', label='Predictions')
plt.xlabel('Input')
plt.ylabel('Output')
plt.legend()
plt.show()

# Evaluate the model on a test set
test_house_sizes = np.array([[1550.0], [1750.0], [1950.0]], dtype=np.float32)
test_actual_prices = np.array([[220000], [240000], [260000]], dtype=np.float32)

# Normalize the test data using the scalers from the training data
test_house_sizes_normalized = house_sizes_scaler.transform(test_house_sizes)
test_actual_prices_normalized = actual_prices_scaler.transform(
    test_actual_prices)

test_loss = model.evaluate(test_house_sizes_normalized,
                           test_actual_prices_normalized)
print(f"Test Loss: {test_loss}")
