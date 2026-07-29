import torch

def main():
    # Check if CUDA is available and set the device accordingly
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create a simple tensor and move it to the selected device
    tensor = torch.tensor([1.0, 2.0, 3.0], device=device)
    print(f"Tensor on {device}: {tensor}")

if __name__ == "__main__":
    main()
