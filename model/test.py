import torch
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms 
import numpy as np
from model import CNN
from train import Data
from tqdm import tqdm


x_test=np.load("../Data/numpy/x_test.npy")
y_test=np.load("../Data/numpy/y_test.npy")
test_transforms=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.55566122, 0.5696501 , 0.53595315],std=[0.29024144, 0.28019984, 0.31316953])
])
testdata=Data(x_test,y_test,test_transforms)
testdataloader=DataLoader(
    dataset=testdata,
    batch_size=32,
    drop_last=True,
    shuffle=False
)

model=CNN()
model.load_state_dict(torch.load("best_model.pth"))
def predicts():
    model.eval()
    device=torch.device("cpu")
    correct=0
    total=0
    with torch.no_grad():
        for batch,(x,y) in tqdm(enumerate(testdataloader),total=len(testdataloader)):
            x=x.float().to(device)
            y=y.float().to(device)
            pred=model(x)
            output=torch.round(pred)
            correct+=(output==y).sum().item()
            total+=len(y)
    test_acc=correct/total*100
    print(f"Test Accuracy : {test_acc:.2f}%")
    
if __name__=="__main__":
    predicts()