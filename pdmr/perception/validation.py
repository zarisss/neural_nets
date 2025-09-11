import torch

def validate(model, Test_load, DEVICE):
    correct, total = 0, 0
    model.eval()
    with torch.no_grad():
        for images, labels in Test_load:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    acc = 100 * correct / total
    print(f"Validation Accuracy: {acc:.2f}%")
    return acc
