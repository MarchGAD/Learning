import torch.nn as nn
import torch

torch.random.manual_seed(1997)
mod = nn.Sequential(
    nn.Linear(in_features=10, out_features=10),
    nn.Dropout(),
    nn.Linear(in_features=10, out_features=1)
)
input = torch.rand(1, 10)

ans1 = mod(input)

ans2 = mod(input)
print('ans1 and ans2 are different, since the dropout layer is working. ans1 - ans2:', (ans1 - ans2).item())
mod.eval()
ans3 = mod(input)

ans4 = mod(input)
print('however, when use .eval, the dropout layer stop, and the difference becomes:',(ans3 - ans4).item())
