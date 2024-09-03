import torch
from torch import nn

class FoodFinder(nn.Module):
  def __init__(self,input_size:int, hidden:int, output_size:int, *args, **kwargs) -> None:
     super().__init__(*args, **kwargs)
     self.conv_layer1 = nn.Sequential(
         nn.Conv2d(in_channels=input_size,out_channels=hidden,kernel_size=3,padding=1,stride=1),
         nn.ReLU(),
         nn.Conv2d(in_channels=hidden,out_channels=hidden,kernel_size=3,padding=1,stride=1),
         nn.ReLU(),
         nn.MaxPool2d(kernel_size=2,stride=2)
         )
     self.conv_layer2 = nn.Sequential(
         nn.Conv2d(in_channels=hidden,out_channels=hidden,kernel_size=3,padding=1,stride=1),
         nn.ReLU(),
         nn.Conv2d(in_channels=hidden,out_channels=hidden,kernel_size=3,padding=1,stride=1),
         nn.ReLU(),
         nn.MaxPool2d(kernel_size=2)
         )
     self.linear_layer = nn.Sequential(
         nn.Flatten(),
         nn.Linear(in_features=hidden*16*16,
                      out_features=output_size)
     )

  def forward(self, x:torch.tensor):
    # x = self.conv_layer1(x)
    # print(x.shape)
    # x = self.conv_layer2(x)
    # print(x.shape)
    # x = self.linear_layer(x)
    # print(x.shape)
    # return x
    return self.linear_layer(self.conv_layer2(self.conv_layer1(x)))
    
