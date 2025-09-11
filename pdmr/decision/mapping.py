# decision/mapper.py
import torch
import torch.nn as nn

# Rule-based mapping: quick demo for review
def rule_map(p_pothole):
    # p_pothole in [0,1] -> v_scale in [0.3,1.0], comfort weight scaled [1,10]
    v_scale = 1.0 - 0.7 * float(p_pothole)    # 0->1.0, 1->0.3
    comfort = 1.0 + 9.0 * float(p_pothole)    # 0->1.0, 1->10.0
    return {"v_scale": v_scale, "comfort": comfort}

# Optional: tiny NN placeholder to replace rule later
class ParamMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 16), nn.ReLU(),
            nn.Linear(16, 2)
        )
    def forward(self, p):
        out = self.net(p.unsqueeze(1))   # p: (B,)
        vscale = torch.sigmoid(out[:,0]) * 1.0
        comfort = 1.0 + torch.relu(out[:,1]) * 10.0
        return vscale, comfort
