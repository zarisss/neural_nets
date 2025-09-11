import matplotlib.pyplot as plt
import numpy as np

def imshow(img, title=""):
    """
    Show a single image tensor.
    """
    img = img.cpu().numpy()
    img = np.transpose(img, (1, 2, 0))
    img = (img * 0.5) + 0.5  # unnormalize
    img = np.clip(img, 0, 1)
    plt.imshow(img)
    if title:
        plt.title(title)
    plt.axis("off")
    plt.show()
