import torch 
from torchvision import transforms
from torch.utils.data import Dataset,DataLoader
from torch import nn,optim
import numpy as np
from model import CNN,TransferLearningModel
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,classification_report,confusion_matrix
from tqdm import tqdm 


model=CNN()

training_train_transforms=transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(20),
        transforms.RandomGrayscale(p=0.2),
        transforms.Normalize(mean=[0.55566122, 0.5696501 , 0.53595315],std=[0.29024144, 0.28019984, 0.31316953])
    ]
)

training_validation_transforms=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.55566122, 0.5696501 , 0.53595315],std=[0.29024144, 0.28019984, 0.31316953])
])

class Data(Dataset):
    def __init__(self,image,label,transforms=None):
        super().__init__()
        self.image=image
        self.label=torch.tensor(label)
        self.transforms=transforms
    
    def __getitem__(self,index):
        label=self.label[index]
        if self.transforms:
            img=self.transforms(self.image[index])
        return (img,label)
    
    def __len__(self):
        return len(self.image)
    
    
x_train=np.load("../Data/numpy/x_train.npy")
y_train=np.load("../Data/numpy/y_train.npy")
x_val=np.load("../Data/numpy/x_val.npy")
y_val=np.load("../Data/numpy/y_val.npy")

traindata=Data(x_train,y_train,training_train_transforms)
testdata=Data(x_val,y_val,training_validation_transforms)

traindataloader=DataLoader(
    dataset=traindata,
    batch_size=32,
    drop_last=False,
    shuffle=True,
)
testdataloader=DataLoader(
    dataset=testdata,
    batch_size=32,
    drop_last=True,
    shuffle=False
)


loss_fn=nn.BCELoss()
optimizer=optim.Adam(model.parameters(),lr=0.001)
device=torch.device("cpu")

def predict():
    correct,total=0,0
    model.eval()
    test_epoch_loss=[]
    all_preds = [] 
    all_labels = []
    with torch.no_grad():
        for batch,(x,y) in tqdm(enumerate(testdataloader),total=len(testdataloader)):
            x=x.float().to(device)
            y=y.float().to(device)
            pred=model(x)
            output=torch.round(pred)
            correct+=(output==y).sum().item()
            total+=len(y)
            all_preds.extend(output.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            loss=loss_fn(pred,y)
            test_epoch_loss.append(loss.item())
    test_acc=(correct/total)*100
    test_loss=np.mean(test_epoch_loss)
    conf_matrix = confusion_matrix(all_labels, all_preds) 
    class_report = classification_report(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    print(f"\nConfusion Matrix:\n{conf_matrix}")
    print(f"Classification Report:\n{class_report}")
    return  test_acc,precision,recall,f1,test_loss
            

def train():        
    epoch=15
    model.to(device)
    avg_train_loss=[]
    avg_test_loss=[]
    test_loss=[]
    test_acc=[]
    best_accuracy = 0 
    patience = 3 
    counter = 0
    for i in range(epoch):
        model.train()
        correct=0
        total=0
        epoch_loss=[]
        for batch,(x,y) in tqdm(enumerate(traindataloader),total=len(traindataloader)):
            x=x.float().to(device)
            y=y.float().to(device)
            optimizer.zero_grad()
            pred=model(x)
            loss=loss_fn(pred,y)
            loss.backward()
            optimizer.step()
            total+=len(y)
            output=(pred>0.5).float()
            correct+=(output==y).sum().item()
            epoch_loss.append(loss.item())
        test_acc,precision,recall,f1,test_loss=predict()
        avg_train_loss=np.mean(epoch_loss)
        train_acc=correct/total*100
        print(f"Epoch [{i+1}/{epoch}] Training_Loss: {avg_train_loss:.4f} Train_Accuracy: {train_acc:.2f}% Testing_Loss: {test_loss:.4f} Test_Accuracy: {test_acc:.2f}%\nPrecision: {precision:.4f} Recall: {recall:.4f} F1-Score: {f1:.4f}")
        #Early stopping implementation
        if test_acc > best_accuracy: 
            best_accuracy = test_acc 
            torch.save(model.state_dict(), "best_model.pth") 
            counter = 0 
            print("Best model saved!") 
        else: 
            counter += 1 
        if counter >= patience: 
            print("Early stopping triggered!") 
            break 

if __name__=="__main__":
    train()