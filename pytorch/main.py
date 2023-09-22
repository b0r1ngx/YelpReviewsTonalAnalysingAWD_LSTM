import torch.backends.mps

device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

mps_device = torch.device(device)

shape = (2, 3,)
# Create tensors directly on MPS device
tensor = torch.rand(
    size=shape,
    device=mps_device
)
print(tensor)
print("on device: ", tensor.device)

# Any operation happens on the GPU
double_tensor = tensor * 2

# Move your model to mps just like any other device
# model = YourFavoriteNet()
# model.to(mps_device)

# Now every call runs on the GPU
# prediction = model(double_tensor)
