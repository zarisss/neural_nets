# integration/demo_integration.py
import numpy as np
import matplotlib.pyplot as plt
import torch
from perception.infer import load_cnn, predict_pothole_prob_batch
from perception.Data_loader import get_data_loaders
from decision.mapping import rule_map
from controller.mpc import solve_bicycle_mpc

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def demo_batch(num_images=17):
    # 1) load perception model & data
    cnn = load_cnn("pothole_cnn.pth")
    _, test_loader, classes = get_data_loaders(Data_directory="Dataset/archive/My Dataset",
                                              image_size=128, batch_size=num_images)
    images, labels = next(iter(test_loader))
    images = images.to(DEVICE)

    probs = predict_pothole_prob_batch(cnn, images)   # shape (B,)
    print("pothole probs:", probs)

    results = []
    for i in range(min(num_images, images.shape[0])):
        p = float(probs[i])
        params = rule_map(p)
        v_scale = params["v_scale"]
        v_max = 5.0 * v_scale   # nominal 5 m/s scaled by perception
        print(f"Img {i}: p={p:.2f} -> v_max={v_max:.2f}")

        # simple initial & target states
        x0 = np.array([0.0, 0.0, 0.0, 0.0])
        xs = np.array([10.0, 0.0, 0.0, 0.0])
        Xopt, Uopt = solve_bicycle_mpc(x0, xs, v_max=v_max, N=12, dt=0.1)

        if Xopt is not None:
            results.append((Xopt, Uopt, v_max))

    # Plot
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    for Xopt, _, v in results:
        plt.plot(Xopt[0,:], Xopt[1,:], '-o', label=f"vmax={v:.2f}")
    plt.xlabel("X (m)"); plt.ylabel("Y (m)"); plt.title("Trajectories"); plt.legend(); plt.grid()

    plt.subplot(1,2,2)
    for Xopt, _, v in results:
        plt.plot(Xopt[3,:], '-o', label=f"vmax={v:.2f}")
    plt.xlabel("step"); plt.ylabel("v (m/s)"); plt.title("Velocity profiles"); plt.legend(); plt.grid()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    demo_batch(num_images=17)
 