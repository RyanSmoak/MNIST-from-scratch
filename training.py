import numpy as np
from tensorflow.keras.datasets import mnist
from cnn import CNN
from SGD_Nestrov import SGD_NAG
from loss import losses

def load_mnist_data():
    """
    Load MNIST dataset and preprocess it.
    :return: Tuple of (train_data, train_labels, test_data, test_labels)
    """
    # Load MNIST data from tensorflow.keras
    (train_data, train_labels), (test_data, test_labels) = mnist.load_data()

    # Normalize the data to be in range [0, 1]
    train_data = train_data.astype(np.float32) / 255.0
    test_data = test_data.astype(np.float32) / 255.0

    # Reshape the data to have shape (batch_size, channels, height, width)
    train_data = train_data.reshape(-1, 1, 28, 28)
    test_data = test_data.reshape(-1, 1, 28, 28)

    # One-hot encode the labels
    train_labels = np.eye(10)[train_labels]  # One-hot encoding for 10 classes
    test_labels = np.eye(10)[test_labels]

    return train_data, train_labels, test_data, test_labels

def train(model, train_data, train_labels, epochs, batch_size, learning_rate, optimizer):
    """
    Train the CNN model using the provided training data.

    :param model: The CNN model instance.
    :param train_data: Training data (images).
    :param train_labels: Training labels (one-hot encoded).
    :param epochs: Number of epochs to train.
    :param batch_size: Size of each training batch.
    :param learning_rate: Learning rate for parameter updates.
    :param optimizer: Optimizer instance (e.g., SGD or SGD with Nesterov momentum).
    """
    num_samples = train_data.shape[0]
    num_batches = num_samples // batch_size

    for epoch in range(epochs):
        epoch_loss = 0
        epoch_accuracy = 0

        # Shuffle data at the beginning of each epoch
        perm = np.random.permutation(num_samples)
        train_data = train_data[perm]
        train_labels = train_labels[perm]

        for i in range(num_batches):
            # Get the current batch
            batch_start = i * batch_size
            batch_end = (i + 1) * batch_size
            x_batch = train_data[batch_start:batch_end]
            y_batch = train_labels[batch_start:batch_end]

            # Forward pass: compute output and loss
            y_pred = model.forward(x_batch).transpose()
            loss_instance = losses(y_batch, y_pred)
            loss = loss_instance.cross_entropy()
            epoch_loss += loss

            # Compute accuracy for this batch
            batch_accuracy = np.mean(np.argmax(y_pred, axis=1) == np.argmax(y_batch, axis=1))
            epoch_accuracy += batch_accuracy

            # Backward pass: compute gradients and update weights
            d_loss = y_pred - y_batch  # For cross-entropy + softmax
            model.backward(d_loss, learning_rate, optimizer)


        # Average loss and accuracy for the epoch
        epoch_loss /= num_batches
        epoch_accuracy /= num_batches

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")

# Example of training loop setup
if __name__ == "__main__":
    # Assuming you have a function to load MNIST dataset
    train_data, train_labels, test_data, test_labels = load_mnist_data()

    # Initialize the model and optimizer
    cnn_model = CNN()
    optimizer_conv1 = SGD_NAG(learning_rate=0.01, momentum=0.9)
    optimizer_conv2 = SGD_NAG(learning_rate=0.01, momentum=0.9)

    # Train the model
    train(cnn_model, train_data, train_labels, epochs=10, batch_size=64,
          learning_rate=0.01,
          optimizer=[optimizer_conv1, optimizer_conv2])
