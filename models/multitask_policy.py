import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiTaskPolicy(nn.Module):
    """
    Outputs:
      - actions: (batch, action_dim)
      - phase_logits: (batch, num_phases)
    """
    def __init__(self, state_dim: int, action_dim: int, num_phases: int = 6,
                 hidden: int = 256, phase_hidden: int = 128):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.action_head = nn.Linear(hidden, action_dim)
        self.phase_head = nn.Sequential(
            nn.Linear(hidden, phase_hidden),
            nn.ReLU(),
            nn.Linear(phase_hidden, num_phases)
        )

    def forward(self, state):
        h = self.trunk(state)
        action = self.action_head(h)
        phase_logits = self.phase_head(h)
        return action, phase_logits