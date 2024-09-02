import torch   # Importing PyTorch library to build the neural network.
import torch.nn as nn    # Importing neural network modules for building and training the model.
import torch.optim as optim   # Importing optimization algorithms for training the neural network.
from torch.utils.data import DataLoader    # Importing DataLoader to handle batching and shuffling of the dataset.
from torchvision import datasets, transforms    # Importing datasets and transforms from torchvision to handle image data and apply transformations.
from torchvision.models import efficientnet_b0   # Importing a pre-trained EfficientNet-B0 model from torchvision.
from PIL import Image    # Importing the PIL library to handle image loading and processing.
from sklearn.metrics import classification_report, confusion_matrix   # Importing metrics for evaluating the model's performance.
import seaborn as sns    # Importing seaborn for visualizing the confusion matrix.
import matplotlib.pyplot as plt   # Importing matplotlib for plotting the confusion matrix.

# Hyperparameters
num_classes = 9   # Number of output classes for the classification task.
batch_size = 50   # Number of images to process in one batch.
learning_rate = 0.0001   # Learning rate for the optimizer.
num_epochs = 20   # Number of epochs to train the model.

# Specify device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')    # Determines whether to use GPU (cuda) or CPU for computations.

# Data transforms with more augmentation
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),   # Resizes the images to 224x224 pixels.
        transforms.RandomHorizontalFlip(),    # Randomly flips the images horizontally to augment the data.
        transforms.RandomRotation(15),   # Randomly rotates the image within a 15-degree range.
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),   # Randomly crops the image and resizes it to 224x224 pixels.
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),   # Randomly changes brightness, contrast, saturation, and hue.
        transforms.ToTensor(),    # Converts the image to a PyTorch tensor.
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])   # Normalizes the image using pre-defined mean and std values.
    ]),
    'test': transforms.Compose([
        transforms.Resize((224, 224)),   # Resizes the image to 224x224 pixels.
        transforms.CenterCrop(224),   # Crops the central 224x224 region of the image.
        transforms.ToTensor(),    # Converts the image to a tensor.
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])    # Normalizes the image with the same values used in training.
    ]),
}

# Datasets and loaders
train_dataset = datasets.ImageFolder(root='C:/Users/sayan/OneDrive/Desktop/testing/Skin cancer ISIC The International Skin Imaging Collaboration/Train', transform=data_transforms['train'])   # Loading the training dataset from the specified directory.
test_dataset = datasets.ImageFolder(root='C:/Users/sayan/OneDrive/Desktop/testing/Skin cancer ISIC The International Skin Imaging Collaboration/Test', transform=data_transforms['test'])   # Loading the testing dataset from the specified directory.

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)   # Creating a DataLoader for the training dataset with batching and shuffling.
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)   # Creating a DataLoader for the testing dataset without shuffling.

# Model
model = efficientnet_b0(weights='DEFAULT')   # Loading the pre-trained EfficientNet-B0 model with default weights (trained on ImageNet).
model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)   # Replacing the last layer of the classifier to match the number of classes in our dataset.
model = model.to(device)   # Transferring the model to the specified device (GPU or CPU).

# Loss and optimizer
criterion = nn.CrossEntropyLoss()   # Defining the loss function for multi-class classification.
optimizer = optim.Adam(model.parameters(), lr=learning_rate)   # Defining the Adam optimizer with the learning rate.

# Training
for epoch in range(num_epochs):   # Looping through the training process for a set number of epochs.
    model.train()   # Setting the model to training mode.
    running_loss = 0.0   # Initializing a variable to accumulate the total loss over the training epoch.
    correct = 0   # Initializing a variable to count the number of correctly classified samples.
    total = 0   # Initializing a variable to track the total number of samples processed.

    for images, labels in train_loader:   # Looping through the batches of images and their corresponding labels from the training data.
        images, labels = images.to(device), labels.to(device)   # Moving the images and labels to the GPU or CPU.

        # Forward pass
        outputs = model(images)   # Forward pass through the model.
        loss = criterion(outputs, labels)   # Calculating the loss between the model’s predictions and the actual labels.

        # Backward and optimize
        optimizer.zero_grad()   # Resetting the gradients to zero before backpropagation.
        loss.backward()   # Backpropagating the loss to compute gradients.
        optimizer.step()   # Updating the model parameters based on the gradients.

        # Statistics
        running_loss += loss.item()   # Accumulating the loss for this batch.
        _, predicted = torch.max(outputs.data, 1)   # Getting the predicted class with the highest probability.
        total += labels.size(0)   # Incrementing the total number of samples.
        correct += (predicted == labels).sum().item()   # Incrementing the count of correct predictions.

    train_accuracy = 100 * correct / total   # Calculating the training accuracy for this epoch.
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}, Accuracy: {train_accuracy:.2f}%')   # Printing the loss and accuracy for this epoch.

# Testing
model.eval()   # Setting the model to evaluation mode.
correct = 0   # Initializing a variable to count the number of correctly classified samples.
total = 0   # Initializing a variable to track the total number of samples processed.
all_labels = []   # Initializing a list to store all true labels.
all_predictions = []   # Initializing a list to store all predicted labels.

class_names = train_dataset.classes   # Getting the class names from the training dataset.

with torch.no_grad():   # Disabling gradient calculation since we are in evaluation mode.
    for images, labels in test_loader:   # Looping through the batches of images and labels from the test data.
        images, labels = images.to(device), labels.to(device)   # Moving the images and labels to the GPU or CPU.
        outputs = model(images)   # Forward pass through the model.
        _, predicted = torch.max(outputs.data, 1)   # Getting the predicted class with the highest probability.
        total += labels.size(0)   # Incrementing the total number of samples.
        correct += (predicted == labels).sum().item()   # Incrementing the count of correct predictions.

        all_labels.extend(labels.cpu().numpy())   # Storing the true labels.
        all_predictions.extend(predicted.cpu().numpy())   # Storing the predicted labels.

test_accuracy = 100 * correct / total   # Calculating the test accuracy.
print(f'Test Accuracy: {test_accuracy:.2f}%')   # Printing the test accuracy.

# Confusion Matrix
'''cm = confusion_matrix(all_labels, all_predictions)  # Compute the confusion matrix.
plt.figure(figsize=(8, 6))  # Create a new figure for the confusion matrix plot.
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)  # Create a heatmap of the confusion matrix with annotations.
plt.xlabel('Predicted')  # Label the x-axis as 'Predicted'.
plt.ylabel('Actual')  # Label the y-axis as 'Actual'.
plt.title('Confusion Matrix')  # Add a title to the plot.
plt.show()  # Display the plot.'''

# Prediction loop for user input
import torch.nn.functional as F  # Importing functional module for softmax operation.

# Set a confidence threshold
confidence_threshold = 0.8  # Setting the confidence threshold to 80%.

def predict_image(image_path):
    image = Image.open(image_path)   # Opening the image from the specified path.
    image = data_transforms['test'](image).unsqueeze(0).to(device)   # Applying the test transforms to the image and adding a batch dimension.

    model.eval()   # Setting the model to evaluation mode.
    with torch.no_grad():   # Disabling gradient calculation since we are making predictions.
        outputs = model(image)   # Forward pass through the model.
        probabilities = F.softmax(outputs, dim=1)   # Applying softmax to get probabilities for each class.
        confidence, predicted = torch.max(probabilities, 1)   # Getting the highest probability and its corresponding class index.
        predicted_class = class_names[predicted[0]]   # Mapping the class index to the actual class name.
        confidence_percentage = confidence.item() * 100   # Converting the confidence to a percentage.

        if confidence_percentage >= confidence_threshold * 100:   # Checking if the confidence is above the threshold.
            return predicted_class, confidence_percentage   # Returning the predicted class and confidence.
        else:
            return "Uncertain", confidence_percentage   # Returning "Uncertain" if confidence is below the threshold.

while True:
    image_path = input("Enter the path of the image to predict (or type 'exit' to quit): ")   # Prompting the user to input the image path.
    if image_path.lower() == 'exit':   # Exiting the loop if the user types 'exit'.
        break

    try:
        predicted_class, confidence_percentage = predict_image(image_path)   # Predicting the class and confidence for the input image.
        if predicted_class == "Uncertain":   # Checking if the prediction is uncertain.
            print(f'Prediction is uncertain with Confidence: {confidence_percentage:.2f}%. No class predicted.')   # Informing the user that the prediction is uncertain.
        else:
            print(f'Predicted Class: {predicted_class} with Confidence: {confidence_percentage:.2f}%')   # Printing the predicted class and confidence.
    except Exception as e:   # Handling any errors that occur during prediction.
        print(f'Error: {e}')   # Printing the error message.
