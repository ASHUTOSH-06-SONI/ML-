import torch
import matplotlib.pyplot as plt
#single starting point
T = 100
beta = 0.02
x0 = torch.tensor(5.0)
xt = x0
trajectory = [xt.item()]
# Forward noising process
for t in range(T):
    noise = torch.randn(())
    xt = (
        torch.sqrt(torch.tensor(1-beta))*xt
    + torch.sqrt(torch.tensor(beta))*noise
    )
    trajectory.append(xt.item())

plt.plot(trajectory)
plt.title("Forward Noising Process")
plt.xlabel("Time Step")
plt.ylabel("x_t")
plt.show()