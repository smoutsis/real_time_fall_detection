import torch
import torch.nn as nn  

#  The architect of the MLP, fc1 input is changed based to the input (3 or 4)
class fc(nn.Sequential):
    def __init__(self, ):
        super(fc, self).__init__()
        
        self.fc1 = nn.Linear(3,64)
        self.act1 = nn.ReLU()
        # self.bn1 = nn.BatchNorm1d(50, affine=False)
        
        self.fc2 = nn.Linear(64,64)
        self.act2 = nn.ReLU()
        # self.bn2 = nn.BatchNorm1d(50, affine=False)
        
        self.fc3 = nn.Linear(64,2)

        
    def forward(self, out):
        
        out = self.fc1(out)
        out = self.act1(out)
        # out = self.bn1(out)
        
        out = self.fc2(out)
        out = self.act2(out)
        # out = self.bn2(out)
        
        out = self.fc3(out)
        
        return out


#  Check if GPU is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


path = 'path to the model'
# Load the pt model
model_fall = torch.load(path)