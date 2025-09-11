import torch
from perception.Data_loader import get_data_loaders
from perception.training import build_model, training
from perception.validation import validate

train_load, test_load, classes = get_data_loaders(Data_directory="Dataset/archive/My Dataset", image_size=128, batch_size=30)
model, criteria, optimiser = build_model(DEVICE='cuda', LR=0.001)

model = training(model, optimiser, criteria, train_load, 'cuda', Epochs=10)
acc = validate(model, test_load, 'cuda')
torch.save(model.state_dict(), "pothole_cnn.pth")