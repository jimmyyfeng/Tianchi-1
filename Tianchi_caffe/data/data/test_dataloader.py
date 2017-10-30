from dataset import Dataset
from dataloader import DataLoader
import numpy as np
class MyDataset(Dataset):
    def __init__(self,root):
        self.data = range(10)

    def __getitem__(self,index):
        return np.array(self.data[index])

    def __len__(self):
        return 10
dataset = MyDataset(None)

dataloader = DataLoader(dataset,batch_size=2,shuffle=True,num_workers=4)

for ii,d in iter(enumerate(dataloader)):
    print ii,d