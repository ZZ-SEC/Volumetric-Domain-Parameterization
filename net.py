import torch
from torch import nn

activation_func = lambda: nn.ReLU(inplace=True)


class ResModule(nn.Module):
    def __init__(self, fn, fn2=lambda x: x):
        super().__init__()
        self.fn = fn
        self.fn2 = fn2

    def forward(self, x):
        return self.fn(x) + self.fn2(x)


class PositionalEncoding(nn.Module):
    def __init__(self, N):
        super().__init__()
        self.N = N
        self.scale = torch.nn.Parameter(torch.tensor(3.14, dtype=torch.float32), requires_grad=True)

    def forward(self, input):
        shape = input.shape
        dim_in = shape[-1]
        input = input.reshape([-1, dim_in])
        # x, y, z = input[:, -3:-2], input[:, -2:-1], input[:, -1:]
        N = self.N
        dim_out = 2 * dim_in * self.N + dim_in
        c = (torch.arange(1, N + 1, device=input.device).reshape([1, 1, -1])) * self.scale
        input_N = (input.reshape([-1, dim_in, 1]) * c)
        sin = (torch.sin(input_N) / c).reshape([-1, dim_in * N])
        cos = (torch.cos(input_N) / c).reshape([-1, dim_in * N])
        # out = torch.cat([input, x ** 2, y ** 2, z ** 2, x * y, x * z, y * z, cos, sin], dim=1)
        out = torch.cat([input, cos, sin], dim=1)
        shape_out = [*shape[:-1], dim_out]
        return out.reshape(shape_out)


class Net(nn.Module):
    def __init__(self, dim_in=3, N_positional_encoding=10, N_hidden=512, depth=6):
        super().__init__()
        dim_encoding = 2 * dim_in * N_positional_encoding + dim_in
        self.N_hidden = N_hidden
        self.net = nn.Sequential(
            # Rotation3(),
            PositionalEncoding(N_positional_encoding),
            nn.Linear(dim_encoding, N_hidden),
            activation_func(),
            *[ResModule(nn.Sequential(
                nn.Linear(N_hidden, N_hidden),
                # nn.LayerNorm(N_hidden),
                # nn.BatchNorm1d(N_hidden),
                activation_func(),
            )) for i in range(depth)],
            nn.Linear(N_hidden, 3),
            # Rotation3(),
        )

    def forward(self, x):
        shape = x.shape
        output = self.net(x.reshape([-1, 3]))
        return output.reshape(shape)

    def initialize(self, save_path="./identity.pth"):
        device = list(self.parameters())[0].device
        optim = torch.optim.Adam(self.parameters(), lr=0.001)
        for i in range(500):
            input = (torch.rand([20000, 3], device=device) * 5 - 2)
            output = self(input)
            loss = torch.nn.functional.mse_loss(input, output)
            optim.zero_grad()
            loss.backward()
            optim.step()
            print(loss.item())
        torch.save(self.state_dict(), save_path)
