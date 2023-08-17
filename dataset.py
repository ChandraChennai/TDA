import os
import cv2
import sys
from torch.utils.data import Dataset
import torch
from torch.nn.functional import normalize
from torchvision import transforms


path = "Data_TPS_PS.txt"
pathImg = "Final_3"
with open(path, "r") as f:
    a = f.readlines()


labelDict = {}

for line in a:
    l = eval(line)
    name = l[0]
    data = l[1:]
    labelDict[name] = data


## Preprocessing code

class ImageVector(Dataset):
    def __init__(self, path: str, transform=None, labelDict=labelDict) -> None:
        super().__init__()

        self.imgPaths = [os.path.join(path, i) for i in os.listdir(path)]
        self.labelDict = labelDict
        self.transform = transform

    def __len__(self) -> int:
        return len(self.imgPaths)
    
    def __getitem__(self, index):
        path = self.imgPaths[index]
        name = path.split("/")[-1].split(".")[0]
    
        img = cv2.imread(path)
        if self.transform:
            img = self.transform(img)

        label = torch.tensor(labelDict[name], dtype=torch.float32)

        return (img.to(torch.float32), label[[ 4, 3, 0]]), label[7]
        #return (img.to(torch.float32), label[:7], label[7] 


class OnlyVector(Dataset):
    def __init__(self, path: str,labelDict=labelDict) -> None:
        super().__init__()

        self.imgPaths = [os.path.join(path, i) for i in os.listdir(path)]
        self.labelDict = labelDict

    def __len__(self) -> int:
        assert len(self.labelDict) == len(self.imgPaths)
        return len(self.imgPaths)

    def __getitem__(self, index) -> torch.Tensor:
        path = self.imgPaths[index]
        name = path.split("/")[-1].split(".")[0]
 
        label = torch.tensor(labelDict[name], dtype=torch.float32)
        #return label[:7], label[7]  
        return label[[4, 3, 0]], label[7]   


if __name__ == "__main__":
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),transforms.Resize((224, 224))])
    data = OnlyVector(pathImg)
    print(data[0])
    
