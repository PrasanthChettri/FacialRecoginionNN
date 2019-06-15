import torch
from torch import nn

class NN(nn.Module):
    def __init__(self , dimensions = (480, 640) , batch_size = 10 ):
        super().__init__()
        self.dimensions = dimensions
        self.batch_size= batch_size 
        #after first layer : dim(1 , 477 , 637)
        self.c1 = nn.Conv2d(1 , 3 , 3 , stride = 1)
        #after secondlayer : dim (1 , 55 , 75) 
        self.c2 = nn.Conv2d(3 , 6 , 5 , stride  = 1) 
        #after third layer : dim (1 , 9 , 13)
        self.c3 = nn.Conv2d(6 , 12 , 3 , stride  = 1)

        #after maxpool :dim (1, 59 , 79)
        self.maxpool_i  = nn.MaxPool2d(8 , 8)
        #after maxpool :dim (1 , 11 , 15)
        self.maxpool_j = nn.MaxPool2d(5,  5)
        #after maxpool :dim (1 , 3 , 4)
        self.maxpool_k = nn.MaxPool2d(3 , 3)


        #Linear Layers
        self.l1 = nn.Linear(144 , 30)
        self.l2 = nn.Linear(30 , 6)
        self.l3  = nn.Linear(6  , 1)

        self.relu = nn.ReLU()
        self.sig = nn.Sigmoid()
        
    def forward(self ,x ):
        x  = self.relu(self.maxpool_i(self.c1(x)))
        x  = self.relu(self.maxpool_j(self.c2(x)))
        x =  self.relu(self.maxpool_k(self.c3(x)))
        x = x.view(self.batch_size , -1)
        x = self.relu(self.l1(x))
        x = self.relu(self.l2(x))
        x = self.sig(self.l3(x))

        return x
        
