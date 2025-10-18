import torch
from torch import nn
from math import pi
import matplotlib.pyplot as plt

# Set up data
r = torch.arange(0, 3, 0.2).unsqueeze(1)  # radii from 0–3
y = pi * (r ** 2)  # true area

# Split into train/test
trainsplit = int(0.7 * len(r))
rtrain, ytrain = r[:trainsplit], y[:trainsplit]
rtest, ytest = r[trainsplit:], y[trainsplit:]

# Define model
class CircleAreaPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1, 8), nn.ReLU(), nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 1))

    def forward(self, x):
        return self.net(x)

# Create instance
torch.manual_seed(0)
model = CircleAreaPredictor()

# Loss & optimizer
lossfn = nn.L1Loss()
optfn = torch.optim.SGD(model.parameters(), lr=0.01)

# Save predictions before training (for visualization)
with torch.inference_mode():
    ypred_before = model(r)

# Training loop
iterations = 2000
for i in range(iterations):
    model.train()
    ypred = model(rtrain)
    loss = lossfn(ypred, ytrain)
    optfn.zero_grad()
    loss.backward()
    optfn.step()

# Predictions after training
with torch.inference_mode():
    ypred_after = model(r)

# visialie
plt.figure(figsize=(8, 5))
plt.scatter(r, y, color='blue', label='True area (πr²)', s=60)
plt.plot(r, ypred_before, color='red', linestyle='--', label='Before training')
plt.plot(r, ypred_after, color='green', label='Predicted area (after training)')
plt.xlabel("Radius (r)")
plt.ylabel("Area")
plt.title("Neural Network Learning the Area of a Circle")
plt.legend()
plt.grid(True)
plt.show()


with torch.inference_mode():
    sample_r = torch.tensor([[1.0]])
    predicted_area = model(sample_r).item()
print(f"Predicted area for r=1: {predicted_area:.4f} (true = {pi:.4f})")