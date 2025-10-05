import torch
from models.bc_policy import SimpleBC

def load_bc_policy(path: str, state_dim: int, action_dim: int, device=None):
    if device is None:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = SimpleBC(state_dim, action_dim).to(device)
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt)
    model.eval()
    return model, device
