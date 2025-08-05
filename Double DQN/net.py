import torch 
import torch.nn as nn 

class QNetwork(nn.Module):
    def __init__(self, state_dim : int, action_dim : int, hidden_dim : int = 128):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, 2 * hidden_dim),
            nn.Tanh(),

            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            
            nn.Linear(hidden_dim, action_dim)
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """
        Kaiming Initialization of the weights helper.
        """
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    
    def forward(self, input : torch.Tensor):
        return self.network(input) 