import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from PIL import Image

# Define the transform
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the dataset
train_dataset = torchvision.datasets.ImageFolder(
    r'C:/Users/sayan/OneDrive/Desktop/new python/SKIN DESEASE/HAM10000_images_part_1', 
    transform=transform
)

test_dataset = torchvision.datasets.ImageFolder(
    r"C:/Users/sayan/OneDrive/Desktop/new python/SKIN DESEASE/HAM10000_images_part_2",
    transform=transform
)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define the model
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # Compute the size after the convolutions and pooling layers
        self.fc1 = nn.Linear(16 * 53 * 53, 120)  # Update these dimensions based on your model's output size
        self.fc2 = nn.Linear(120, 80)
        self.fc3 = nn.Linear(80, len(train_dataset.classes))  # Output should match the number of classes

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 53 * 53)  # Adjust this if dimensions do not match
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Initialize the model, loss function, and optimizer
model = CNNModel()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)  # Try different learning rates

# Use GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

# Train the model
for epoch in range(10):
    model.train()
    running_loss = 0.0
    for i, (x, y) in enumerate(train_loader, 0):
        x, y = x.to(device), y.to(device)
        outputs = model(x)
        loss = criterion(outputs, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 10 == 9:  # Print every 10 mini-batches
            print(f'Epoch {epoch+1}, Batch {i+1}, Loss: {running_loss / 10:.4f}')
            running_loss = 0.0

# Evaluate the model on the test set
def evaluate(model, test_loader):
    model.eval()
    total_correct = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == y).sum().item()
    accuracy = (total_correct / len(test_loader.dataset)) * 100
    return accuracy

accuracy = evaluate(model, test_loader)
print(f'Test accuracy: {accuracy:.2f}%')

# Save the model
torch.save(model.state_dict(), 'model.pth')

# Load the model and make predictions
model = CNNModel()
model.load_state_dict(torch.load('model.pth'))
model.to(device)

# Get class names from the training dataset
class_names = train_dataset.classes

def predict(image_path):
    image = Image.open(image_path)
    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension
    image = image.to(device)
    model.eval()
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
    return class_names[predicted.item()]

# Example usage of the predict function
image_path = r'C:/Users/sayan/OneDrive/Desktop/new python/SKIN DESEASE/HAM10000_images_part_1/akiec/ISIC_0024372.jpg'  # Update with the correct path to the image you want to predict
predicted_class = predict(image_path)
print(f'The predicted class for the image is: {predicted_class}')
