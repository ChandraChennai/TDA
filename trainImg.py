import torch
import torch.nn as nn
import torchvision.transforms as transforms
from Model import *
from dataset import *
from argparse import ArgumentParser
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim import Adam
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import MultiStepLR
import matplotlib.pyplot as plt
import numpy as np
from torch.optim import LBFGS



torch.manual_seed(10)
parser = ArgumentParser()

parser.add_argument("--pathImg", "-pi", default="Final_3", type=str)
parser.add_argument("--name", "-n", required=True, type=str)
parser.add_argument("--batch_size", "-bs", default=32, type=int)
parser.add_argument("--epochs", "-e", default=1000, type=int)
parser.add_argument("--learning_rate", "-lr", default=1e-3, type=float)

args = parser.parse_args()


device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),transforms.Resize((224, 224))])

vecData = ImageVector(args.pathImg, transform)
trainVec, testVec = train_test_split(vecData, test_size=0.20)
trainVecLoader, testVecLoader = DataLoader(trainVec, batch_size=args.batch_size, shuffle=True), DataLoader(testVec, batch_size=args.batch_size)

modelVec = ImgVector2(3, 128, 7, 32, 1).to(device) 
lossFn = nn.MSELoss()
optim = Adam(modelVec.parameters(), args.learning_rate)
scheduler = MultiStepLR(optim, milestones=[50, 100, 150],gamma=0.1)

def train():
    print("Training Begins")
    full_val = []
    
    for e in range(1, args.epochs+1):
        modelVec.train()
        Tloss = []
        for (img, x), y in tqdm(trainVecLoader):
            img, x, y = img.to(device), x.to(device), y.to(device)

            optim.zero_grad()

            y_pred = modelVec(img, x).squeeze(1)
            loss = lossFn(y_pred, y)
            
            loss.backward()
            optim.step()
            

        scheduler.step()
        
        with torch.no_grad():
            loss = []
            modelVec.eval()
            for (img, x), y in testVecLoader:
                img, x, y = img.to(device), x.to(device), y.to(device)
                
                y_pred = modelVec(img, x).squeeze(1)
                loss.append(lossFn(y_pred, y).item())
            print(f"The validation Loss after Epoch{e} is {sum(loss)/len(loss):0.5f}", "\n")
            full_val.append(sum(loss)/len(loss))
            
    print(f"Minimum loss {min(full_val), full_val.index(min(full_val))}")
    weight = modelVec.cpu().fc.weight
    weight.requires_grad = False
    weight = weight.numpy()


    fig, (axs1, axs2) = plt.subplots(1, 2)
    
    im1 = axs1.imshow(weight[:, :-3], cmap="bwr")
    plt.colorbar(im1)

    im2 = axs2.imshow(weight[:, -3:], cmap = "bwr")
    plt.colorbar(im2)

    plt.savefig(args.name + '.png')
    plt.show() 
    
    print(np.linalg.norm(weight[:,:-7])/np.linalg.norm(weight))
if __name__ == "__main__":
    train()
