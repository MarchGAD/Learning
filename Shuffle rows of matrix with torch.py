import torch

shuffled = torch.arange(12).reshape(3, 4)
print(shuffled)
shu = shuffled[torch.randperm(shuffled.size(0)), :]
print(shu)