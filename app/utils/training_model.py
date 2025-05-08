import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import torch
import torch.nn as nn
import torch.optim as optim
import os

def download_data(data_dir='data'):
    os.makedirs(data_dir, exist_ok=True)
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    train_data = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform
    )
    val_data = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform
    )
    return train_data, val_data


class BirdFrogClassifier(nn.Module):  # Renamed from CNN
    def __init__(self):
        super(BirdFrogClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32 * 32 * 32, 128)
        self.fc2 = nn.Linear(128, 2)  # Only 2 classes: bird and frog

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 32 * 32 * 32)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Filter CIFAR-10 dataset for specific classes
def filter_cifar10(dataset, target_classes):
    indices = [i for i, label in enumerate(dataset.targets) if label in target_classes]
    return Subset(dataset, indices)

def train_data():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Transformations
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    # Load CIFAR-10
    train_dataset = datasets.CIFAR10(root="data", train=True, download=True, transform=transform)
    val_dataset = datasets.CIFAR10(root="data", train=False, download=True, transform=transform)

    # Filter dataset for bird (2) and frog (6)
    target_classes = [2, 6]
    train_dataset = filter_cifar10(train_dataset, target_classes)
    val_dataset = filter_cifar10(val_dataset, target_classes)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Initialize model, loss function, and optimizer
    model = BirdFrogClassifier().to(device)  # Updated class name
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    for epoch in range(5):  # Training for 5 epochs
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            # Map labels to [0, 1] for bird and frog
            labels = torch.tensor([0 if label == 2 else 1 for label in labels]).to(device)
            images = images.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}")

        # Validate the model
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                labels = torch.tensor([0 if label == 2 else 1 for label in labels]).to(device)
                images = images.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        print(f"Epoch {epoch+1}, Validation Loss: {val_loss/len(val_loader):.4f}")

    # Save the trained model
    os.makedirs("model", exist_ok=True)
    torch.save(model.state_dict(), "model/train_model.pth")
    print("Model saved to model/train_model.pth")


if __name__ == "__main__":
    download_data()
    train_data()