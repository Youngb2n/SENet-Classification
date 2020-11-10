import torch
import torch.nn as nn
import torch.nn.functional as F
import os, glob
from PIL import Image
from torch.utils.data import Dataset,DataLoader
def FilePath(PATH):
    Folder = os.listdir(PATH)

    train_path = []
    val_path = []
    test_path = []
    li =[train_path, val_path, test_path]
    for index, folder in enumerate(Folder):
        folder_path = os.path.join(PATH,folder)
        folder_path_list = os.listdir(folder_path)
        for fp in folder_path_list:
            folder_path2 = os.path.join(folder_path,fp)
            Files = glob.glob(folder_path2+'/*.'+'jpg')
            li[index] += Files
    return li

class cd_Dataset(Dataset):
    def __init__(self ,transform =None, FILE_PATHS= None):
        self.transform = transform
        self.FILE_PATHS = FILE_PATHS
        self.Image_List=[]
        self.Label_List=[]

        #라벨링
        for i in range(len(FILE_PATHS)):
            if 'dog' in FILE_PATHS[i]:
                self.Image_List.append(FILE_PATHS[i])
                self.Label_List.append(1)
            elif 'cat' in FILE_PATHS[i]:
                self.Image_List.append(FILE_PATHS[i])
                self.Label_List.append(0)
    
    def __len__(self):
        return len(self.Label_List)

    def __getitem__(self, idx):
        label = self.Label_List[idx]
        img = Image.open(self.Image_List[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label
