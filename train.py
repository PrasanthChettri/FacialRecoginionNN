import numpy as np
import torchvision
import  torchvision.transforms  as transforms
import torch
from torch import optim  , nn
from model import NN 
class train_nn:
    def __init__(self):
        self.labels = torch.Tensor(np.fromfile('labels.dat', dtype = int)) 
        #CURRENT NO OF IMAGES and LABELS
        self.n_data = len(self.labels)

        transform_i = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5,), (0.5,))])

        train_dataset = torchvision.datasets.ImageFolder(
                        root = 'images/', 
                        transform = transform_i)
        
        self.dataloader = torch.utils.data.DataLoader(
                        list(zip(train_dataset , self.labels)) , 
                        batch_size = 30 , 
                        num_workers = 0 ,
                        shuffle = True)

    def train(self , epochs = 10):
        Model = NN(batch_size = 30)
        opt = optim.Adam(Model.parameters() , lr = 0.005)
        criterion = nn.BCELoss()
        softmax = nn.Softmax(dim = 0)
        loss = 0 
        print (self.labels.shape)
        for i in range(epochs):
            item_loss = 0 
            for  i ,  (feat , lab) in enumerate(self.dataloader):
                feat = feat[0][:, :1 ,: ,:]
                output = Model.forward(feat)
                loss = criterion(output , lab.view((30)))
                loss.backward()
                opt.step()
                item_loss += loss.item()
            print ("LOSS ---->" , item_loss/len(self.dataloader))
        torch.save(Model.state_dict() , "1")
            
if __name__ == '__main__':
    k = train_nn()
    k.train()
