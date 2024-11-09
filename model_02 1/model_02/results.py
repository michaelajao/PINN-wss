import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

###################################### The neural network. \R^3 to \R^4
class NSNeuralNet(nn.Module):
    def __init__(self, width=256):
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(3,width),
            nn.SiLU(),
            nn.Linear(width,width),
            nn.SiLU(),
            nn.Linear(width,width),
            nn.SiLU(),
            nn.Linear(width,width),
            nn.SiLU(),
            nn.Linear(width,width),
            nn.SiLU(),
            nn.Linear(width,width),
            nn.SiLU(),
            nn.Linear(width,width),
            nn.SiLU(),
            nn.Linear(width,width),
            nn.SiLU(),
            nn.Linear(width,width),
            nn.SiLU(),
            nn.Linear(width,width),
            nn.SiLU(),
            nn.Linear(width,width),
            nn.SiLU(),
            nn.Linear(width,width),
            nn.SiLU(),
            nn.Linear(width,4)
        )

    def forward(self, x, y, z):
        return self.main(torch.cat([x, y, z], axis=1))

###################################### SETUP THE NEURAL NETWORK
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = NSNeuralNet().to(device)
model.apply(lambda m: nn.init.kaiming_normal_(m.weight) if type(m) == nn.Linear else None);

checkpoint = torch.load('ns_34_3_02.pt', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])

###################################### EVALUATE THE MODEL & EXPORT
def get_predictions(net, inp):
    x = torch.tensor(inp[:,0], dtype=torch.float, device=device).reshape(-1,1)
    y = torch.tensor(inp[:,1], dtype=torch.float, device=device).reshape(-1,1)
    z = torch.tensor(inp[:,2], dtype=torch.float, device=device).reshape(-1,1)
    return model(x, y, z).detach().numpy()[:,0]

for sc in range(1,4):
    print(f'Case {sc}.')
    
    print('Plane xy:')
    pts = np.loadtxt(f'../data/mesh_{sc}_xy_eval.csv', delimiter=',')
    preds = get_predictions(model, pts).reshape(-1,1)
    np.savetxt(f'p_{sc}_xy.csv', np.concatenate((pts, preds), axis=1), delimiter=',', fmt='%.8e', header='x,y,z,p')

    print('Plane xz:')
    pts = np.loadtxt(f'../data/mesh_{sc}_xz_eval.csv', delimiter=',')
    preds = get_predictions(model, pts).reshape(-1,1)
    np.savetxt(f'p_{sc}_xz.csv', np.concatenate((pts, preds), axis=1), delimiter=',', fmt='%.8e', header='x,y,z,p')
    print()
