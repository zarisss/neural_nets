import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

# Build Model
def build_model(DEVICE='cuda', LR=0.001):
    model = models.mobilenet_v2(weights="IMAGENET1K_V1")
    model.classifier[1] = nn.Linear(model.last_channel, 2)
    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimiser = optim.Adam(model.parameters(), lr=LR)
    return model, criterion, optimiser

# Training Loop
def training(model, optimiser, criterion, Train_load, DEVICE, Epochs=10):
    loss_history = []
    for epoch in range(Epochs):
        model.train()
        running_loss = 0.0
        for images, labels in Train_load:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimiser.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimiser.step()
            running_loss += loss.item()
        
        epoch_loss = running_loss / len(Train_load)
        loss_history.append(epoch_loss)
        print(f"Epoch {epoch+1}/{Epochs}, Loss: {epoch_loss:.4f}")
    
    return model
