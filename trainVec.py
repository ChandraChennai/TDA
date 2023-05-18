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
import math

torch.manual_seed(10)
parser = ArgumentParser()

parser.add_argument("--pathImg", "-pi", default="./Final_3", type=str)
parser.add_argument("--batch_size", "-bs", default=32, type=int)
parser.add_argument("--epochs", "-e", default=1000, type=int)
parser.add_argument("--learning_rate", "-lr", default=1e-3, type=float)

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

vecData = OnlyVector(args.pathImg)


trainVec, testVec = train_test_split(vecData, test_size=0.25)


trainVecLoader, testVecLoader = DataLoader(trainVec, batch_size=args.batch_size, shuffle=True), DataLoader(testVec, batch_size=args.batch_size)

modelVec = SLP(3, 256, 1).to(device)
lossFn = nn.MSELoss()
optim = Adam(modelVec.parameters(), args.learning_rate)
avg_filal = 0


def train():
    print("Training Begins")
    full_val = []
    avg = 0
    for e in range(1, args.epochs+1):
        modelVec.train()
        for (x, y) in tqdm(trainVecLoader):
            x, y = x.to(device), y.to(device)

            optim.zero_grad()

            y_pred = modelVec(x).squeeze(1)
            loss = lossFn(y_pred, y)
   
            loss.backward()
            optim.step()
        with torch.no_grad():
            loss = []
            modelVec.eval()
            total = 0
            correct = 0
            for (x,y) in testVecLoader:
                x, y = x.to(device), y.to(device) 
                y_pred = modelVec(x).squeeze(1)

                y1 = y.flatten()
                y1 = y1.tolist()
                
                y_pred1 = y_pred.flatten()
                y_pred1 = y_pred1.tolist()
                
                total += y.size(0)
                
                loss.append(lossFn(y_pred,y).item())
                
            print(f"The validation Loss in after Epoch { e} is {sum(loss)/len(loss):0.5f}", "\n")
                
        full_val.append(sum(loss)/len(loss))
    
if __name__ == "__main__":
 train()
 
