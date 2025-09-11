import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
def get_data_loaders(Data_directory='location', image_size=128, batch_size=30):
    transform = transforms.Compose([transforms.Resize((image_size, image_size)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
                                    ])
    Train_Dataset = datasets.ImageFolder(f"{Data_directory}/train", transform=transform)
    Test_Dataset = datasets.ImageFolder(f"{Data_directory}/test", transform=transform)

    Train_load = DataLoader(Train_Dataset, batch_size=batch_size, shuffle=True)
    Test_load = DataLoader(Test_Dataset, batch_size=batch_size, shuffle=False)
    
    return Train_load, Test_load, Train_Dataset.classes