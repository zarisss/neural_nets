##this code is a backup code if things go wrong##

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import numpy as np

BATCH_SIZE = 30
IMAGE_SIZE = 128
EPOCHS = 10
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

# dataset
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
])

Train_Data = datasets.ImageFolder("Dataset/archive/My Dataset/train", transform=transform)
Test_Data  = datasets.ImageFolder("Dataset/archive/My Dataset/test", transform=transform)

Train_load = DataLoader(Train_Data, batch_size=BATCH_SIZE, shuffle=True)
Test_load  = DataLoader(Test_Data, batch_size=BATCH_SIZE, shuffle=False)

print("Classes:", Train_Data.classes)

# model
model = models.mobilenet_v2(weights="IMAGENET1K_V1")
model.classifier[1] = nn.Linear(model.last_channel, 2)  # binary classification
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training + Validation
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for images, labels in Train_load:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {running_loss/len(Train_load):.4f}")

    # Validation
    correct, total = 0, 0
    model.eval()
    with torch.no_grad():
        for imgs, labels in Test_load:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    acc = 100 * correct / total
    print(f"Validation Accuracy: {acc:.2f}%")
torch.save(model.state_dict(), "pothole_cnn.pth")

# ----------------------------
# Visual Validation (Sample Images from Test Set)
# ----------------------------
def imshow(img, title):
    img = img / 2 + 0.5  # unnormalize from [-1,1] to [0,1]
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.title(title)
    plt.axis("off")
    plt.show()

# get a batch from test set
dataiter = iter(Test_load)
images, labels = next(dataiter)
images, labels = images.to(DEVICE), labels.to(DEVICE)

# predictions
outputs = model(images)
_, preds = torch.max(outputs, 1)

# show first 6 images with predicted vs actual
for i in range(17):
    imshow(images[i].cpu(), 
           title=f"Predicted: {Train_Data.classes[preds[i]]} | Actual: {Train_Data.classes[labels[i]]}")
