import torch
import torch.nn as nn

class PINN_Ez(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layers)-1):
            self.layers.append(nn.Linear(layers[i], layers[i+1]))
        self.activation = nn.Tanh()

    def forward(self, coords):
        """
        coords: torch.Tensor of shape (N, 2) representing (x,y)
        returns: torch.Tensor of shape (N,1) representing E_z
        """
        x = coords
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        ez = self.layers[-1](x)
        return ez
