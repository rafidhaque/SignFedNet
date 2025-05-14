# data_utils.py
import numpy as np
import torch
from torchvision import datasets
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from model import TrafficSignModel
import matplotlib.pyplot as plt

def load_data(num_clients: int = 5, test_size: float = 0.2):
    """
    Load and partition GTSRB dataset for federated learning
    
    Args:
        num_clients: Number of federated clients
        test_size: Proportion of local test set (per client)
    
    Returns:
        list: [(train_dataset, test_dataset)] for each client
    """
    # Get transforms from model definition
    transforms = TrafficSignModel.get_transform()
    
    # Download full dataset
    full_train = datasets.GTSRB(
        root='./data',
        split='train',
        download=True,
        transform=transforms['train']
    )
    
    # Create client partitions
    client_datasets = []
    client_size = len(full_train) // num_clients
    
    for i in range(num_clients):
        # Split indices for this client
        start_idx = i * client_size
        end_idx = (i+1)*client_size if i < num_clients-1 else len(full_train)
        
        # Create client subset
        client_indices = list(range(start_idx, end_idx))
        client_data = Subset(full_train, client_indices)
        
        # Collect labels for stratification
        client_labels = []
        for idx in client_indices:
            _, label = full_train[idx]
            client_labels.append(label)

        # Use collected labels for stratification
        train_idx, test_idx = train_test_split(
            np.arange(len(client_data)),
            test_size=test_size,
            stratify=client_labels
        )

        
        client_datasets.append((
            Subset(client_data, train_idx),
            Subset(client_data, test_idx)
        ))
    
    return client_datasets

def get_client_dataloader(client_data, batch_size=32, is_train=True):
    """Create DataLoader for a client's dataset"""
    return DataLoader(
        client_data,
        batch_size=batch_size,
        shuffle=is_train,
        pin_memory=True,
        num_workers=2
    )

def visualize_samples(dataset, num_samples=6):
    """Visualize dataset samples"""
    classes = dataset.dataset.classes
    loader = DataLoader(dataset, batch_size=num_samples, shuffle=True)
    images, labels = next(iter(loader))
    
    plt.figure(figsize=(10, 6))
    for i, (img, label) in enumerate(zip(images, labels)):
        # Denormalize for visualization
        img = img.numpy().transpose(1, 2, 0)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = std * img + mean
        img = np.clip(img, 0, 1)
        
        plt.subplot(2, 3, i+1)
        plt.imshow(img)
        plt.title(f"Class: {classes[label.item()]}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Test data loading
    print("Testing data utilities...")
    client_datasets = load_data(num_clients=5)
    
    # Verify splits
    for i, (train, test) in enumerate(client_datasets):
        print(f"Client {i+1}:")
        print(f"  Train samples: {len(train)}")
        print(f"  Test samples: {len(test)}")
    
    # Visualize samples from first client's train set
    print("\nVisualizing samples from client 1:")
    visualize_samples(client_datasets[0][0])
