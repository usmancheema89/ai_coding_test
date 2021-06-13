
from torch import nn 
import torch
import torchvision.models as models

class My_Model(nn.Module):
    def __init__(self,c=17, pre_model = models.mobilenet_v3_small(pretrained=True)):
        super(My_Model, self).__init__()
        op_f = 96
        self.layers = nn.Sequential(*list(pre_model.features)[:-1])
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(.1)
        self.linear = nn.Linear(op_f,c)
        self.classifier = nn.Softmax(1)
    
    def forward(self,x):
        x = self.layers(x)
        x = self.avgpool(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        x = self.classifier(x)
        return x

# input = torch.rand((4,3,224,224))
# model = My_Model()
# output = model(input)

# print(output.size())