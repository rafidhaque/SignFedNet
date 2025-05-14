# client.py
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets
import flwr as fl
from torch.utils.data import DataLoader, random_split

# 1. Model Definition: Transfer learning with ResNet50
class TrafficSignModel(nn.Module):
    def __init__(self, num_classes=43):
        super().__init__()
        self.base_model = models.resnet50(weights="IMAGENET1K_V2")  # Use latest torch weights argument
        self.base_model.fc = nn.Linear(self.base_model.fc.in_features, num_classes)

    def forward(self, x):
        return self.base_model(x)

# 2. Data Loading and Preprocessing
def load_gtsrb_data(data_dir="./data", train=True):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    dataset = datasets.GTSRB(root=data_dir, split="train" if train else "test", download=True, transform=transform)
    return dataset

def get_data_loaders(split_ratio=0.8, batch_size=32):
    full_dataset = load_gtsrb_data()
    train_size = int(split_ratio * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, val_loader

# 3. Flower Client Implementation
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, val_loader, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.model.to(self.device)

    def get_parameters(self, config):
        return [val.cpu().numpy() for val in self.model.parameters()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v).to(self.device) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=config.get("lr", 0.001))
        epochs = config.get("epochs", 1)
        for epoch in range(epochs):
            for inputs, labels in self.train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
        return self.get_parameters({}), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        criterion = nn.CrossEntropyLoss()
        correct, total, loss_sum = 0, 0, 0.0
        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss_sum += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = correct / total
        avg_loss = loss_sum / total
        return float(avg_loss), total, {"accuracy": accuracy}

# 4. Main: Start the Flower client
if __name__ == "__main__":
    # Detect device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data and model
    train_loader, val_loader = get_data_loaders(batch_size=32)
    model = TrafficSignModel(num_classes=43)

    # Start Flower client
    client = FlowerClient(model, train_loader, val_loader, device)
    fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=client)
