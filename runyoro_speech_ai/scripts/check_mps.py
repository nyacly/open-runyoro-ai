import torch

def check_mps_availability():
    print(f"PyTorch version: {torch.__version__}")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("MPS backend is available on this device.")
        try:
            # Test MPS with a simple tensor operation
            tensor_mps = torch.randn(10, 10, device="mps")
            tensor_cpu = tensor_mps.cpu()
            print("Successfully created a tensor on MPS and moved it to CPU.")
            print("MPS appears to be functional.")
            return True
        except Exception as e:
            print(f"MPS is available, but a test operation failed: {e}")
            return False
    elif not hasattr(torch.backends, "mps"):
        print("MPS backend is not available in this PyTorch build (torch.backends.mps does not exist).")
        print("Ensure you have PyTorch version 1.12 or later for official MPS support, or a nightly build for earlier experimental support.")
        return False
    else: # hasattr(torch.backends, "mps") but not torch.backends.mps.is_available()
        print("MPS backend exists in this PyTorch build but is not currently available on this device.")
        print("This might occur if you are not running on an Apple Silicon Mac or if there's an issue with the MPS drivers/runtime.")
        return False

if __name__ == "__main__":
    check_mps_availability()
