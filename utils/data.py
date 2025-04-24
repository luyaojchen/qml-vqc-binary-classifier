import torch
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
import numpy as np

def load_mnist_0_vs_1_resized(size: int = 8) -> tuple:
    """
    Loads and preprocesses the MNIST dataset, filtering for digits 0 and 1,
    resizing each image to size x size (e.g., 8x8 -> 64 features), and 
    optionally balances the training set to have equal 0s and 1s.
    """
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor()
    ])

    dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)

    # Filter only digits 0 and 1
    filtered = [(img, label) for img, label in dataset if label in [0, 1]]
    images, labels = zip(*filtered)

    images = torch.stack(images).view(-1, size * size)
    labels = torch.tensor(labels, dtype=torch.float32)

    # Normalize for amplitude encoding
    norms = torch.norm(images, dim=1, keepdim=True)
    normalized_images = images / norms

    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        normalized_images.numpy(), labels.numpy(), test_size=0.2, random_state=42
    )

    X_train_0 = X_train[y_train == 0]
    X_train_1 = X_train[y_train == 1]

    y_train_0 = y_train[y_train == 0]
    y_train_1 = y_train[y_train == 1]

    n = min(len(X_train_0), len(X_train_1))

    # Randomly select from each
    idx0 = np.random.choice(len(X_train_0), n, replace=False)
    idx1 = np.random.choice(len(X_train_1), n, replace=False)

    X_train_bal = np.vstack((X_train_0[idx0], X_train_1[idx1]))
    y_train_bal = np.concatenate((y_train_0[idx0], y_train_1[idx1]))

    # Shuffle
    perm = np.random.permutation(len(y_train_bal))
    X_train = X_train_bal[perm]
    y_train = y_train_bal[perm]

    return X_train, X_test, y_train, y_test