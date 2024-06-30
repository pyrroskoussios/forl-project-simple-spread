import torch.nn as nn
import torch.nn.functional as F

def WEIGHT_INITIALIZATION(layer):
    if isinstance(layer, nn.Linear):
        nn.init.orthogonal_(layer.weight.data)
        layer.bias.data.fill_(0.0)

class RNN(nn.Module):
    def __init__(self, input_dimension, hidden_dimension, hidden_depth, output_dimension, output_splits=1, output_activation="linear"):
        super().__init__()
        self.rnn_layers = nn.GRU(input_dimension, hidden_dimension, 1 + hidden_depth, batch_first=True)
        self.output_layers = nn.ModuleList([nn.Linear(hidden_dimension, output_dimension) for _ in range(output_splits)])
        self.output_splits = output_splits
        self.output_activation = output_activation
        self.apply(WEIGHT_INITIALIZATION)

    def forward(self, x):
        x, _ = self.rnn_layers(x)
        if x.dim() == 2:
            x = x[-1].squeeze()
        elif x.dim() == 3:
            x = x[:, -1, :].squeeze()

        outputs = []
        for output_layer in self.output_layers:
            output = output_layer(x)
            if self.output_activation == "softmax":
                output = F.softmax(output, dim=1 if output.dim() > 1 else 0)
            outputs.append(output)
        if self.output_splits == 1:
            outputs = outputs[0]
        return outputs

