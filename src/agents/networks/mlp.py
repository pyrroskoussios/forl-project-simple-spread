import torch.nn as nn
import torch.nn.functional as F

def WEIGHT_INITIALIZATION(layer):
    if isinstance(layer, nn.Linear):
        nn.init.orthogonal_(layer.weight.data)
        layer.bias.data.fill_(0.0)

class MLP(nn.Module):
    def __init__(self, input_dimension, hidden_dimension, hidden_depth, output_dimension, output_activation="linear"):
        super().__init__()
        self.input_layer = nn.Linear(input_dimension, hidden_dimension)
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_dimension, hidden_dimension) for _ in range(hidden_depth)])
        self.output_layer = nn.Linear(hidden_dimension, output_dimension) 
        self.output_activation = output_activation
        self.apply(WEIGHT_INITIALIZATION)

    def forward(self, x):
        x = F.relu(self.input_layer(x))
        for hidden_layer in self.hidden_layers:
            x = F.relu(hidden_layer(x))
        x = self.output_layer(x)
        if self.output_activation == "softmax":
            x = F.softmax(x, dim=1 if x.dim() > 1 else 0)
        return x
