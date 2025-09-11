# perception/infer.py
import torch
from torchvision import models, transforms
from PIL import Image

DEVICE = "cpu"

# Define preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def load_cnn(weights_path="pothole_cnn.pth", device=DEVICE):
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = torch.nn.Linear(model.last_channel, 2)  # 2 classes
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

def predict_pothole_prob_batch(model, images):
    # Case 1: already a batched tensor
    if isinstance(images, torch.Tensor):
        imgs_tensor = images.to(DEVICE)

    # Case 2: list of tensors
    elif isinstance(images[0], torch.Tensor):
        imgs_tensor = torch.stack(images).to(DEVICE)

    # Case 3: list of file paths or PIL Images
    else:
        imgs_tensor = torch.stack([
            transform(Image.open(img).convert("RGB")) if isinstance(img, str) else transform(img)
            for img in images
        ]).to(DEVICE)

    with torch.no_grad():
        outputs = model(imgs_tensor)
        probs = torch.softmax(outputs, dim=1)[:, 1]  # pothole probability
    return probs.cpu().numpy()
