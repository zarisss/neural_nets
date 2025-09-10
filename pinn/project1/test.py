import torch
import time

# Ensure we're targeting the GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running stress test on device: {device}")
print("Watch your rocm-smi terminal!")

if device == "cuda":
    # Create very large tensors directly on the GPU
    # This will use a few GB of VRAM
    a = torch.randn(15000, 15000, device=device)
    b = torch.randn(15000, 15000, device=device)
    
    print("\nStarting heavy GPU computation...")
    start_time = time.time()
    
    # Perform 50 large matrix multiplications
    for i in range(50):
        c = torch.matmul(a, b)
        print(f"  - Iteration {i+1}/50 complete")

    # This special command waits for all GPU work to finish
    # It makes our timing more accurate
    torch.cuda.synchronize()
    end_time = time.time()
    
    print(f"\nGPU stress test finished in {end_time - start_time:.4f} seconds.")
else:
    print("GPU not available. Cannot run stress test.")