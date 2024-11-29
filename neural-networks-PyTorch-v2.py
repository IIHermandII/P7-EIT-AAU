import os
import re
import torch
import torch.nn as nn
import torch.nn.functional as func
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder


# print('--- Code Start ---\n')

def GetNewestDataFileName():
    #Check for env variable - error if not present
    envP7RootDir = os.getenv("P7RootDir")
    if envP7RootDir is None:
        print("---> If you are working in vscode\n---> you need to restart the aplication\n---> After you have made a env\n---> for vscode to see it!!")
        print("---> You need to make a env called 'P7RootDir' containing the path to P7 root dir")
        raise ValueError('Environment variable not found (!env)')
    
    #Enter CSV directory
    workDir = envP7RootDir + "\\Data\\CSV files"
    
    #Find all dates from the files
    dirDates = []
    for file in os.listdir(workDir):
        onlyDate = re.findall(r'\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}', file)
        new_string = str(onlyDate).replace("-", "")
        new_string = new_string.replace("_","")
        new_string = new_string.strip("[]'")
        dirDates.append([int(new_string),file])
    
    #Sort dates and return newest
    dirDates = sorted(dirDates,key=lambda l:l[1],reverse=True) # Take oldest data first i believe 
    return(workDir + "\\" + dirDates[0][1])

class CollectDataSet(torch.utils.data.Dataset):
    def __init__(self, values, labels):
        assert len(values) == len(labels)

        self.values = values
        self.labels = labels
    
    def __len__(self):
        return len(self.values)
    
    def __getitem__(self, index):
        assert 0 <= index and index < len(self)
        return (
            torch.tensor(self.values[index], dtype=torch.double), 
            torch.tensor(self.labels[index], dtype=torch.long)
    )

# newestData = GetNewestDataFileName()
def pytorchPackageData(newestData):

    # Load merged dataset
    dataTemp = pd.read_csv(newestData)

    # Remove filename column
    X = dataTemp.drop(['Filename'], axis = 1)
    print('Current file: ', newestData)

    # Keep data and labels
    data = X.iloc[:, 1:].values
    labels = X.iloc[:, 0].values
    print("Data length: ", len(data))

    data_S = StandardScaler().fit_transform(data)

    classes  = sorted(set(labels))

    BS = 64

    x_train, x_test, y_train, y_test = train_test_split(data_S, labels, test_size = 0.2, random_state=420)

    label_encoder = LabelEncoder()

    y_train_int = label_encoder.fit_transform(y_train)
    y_test_int = label_encoder.fit_transform(y_test)

    train_set = CollectDataSet(x_train, y_train_int)
    test_set = CollectDataSet(x_test, y_test_int)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=BS, pin_memory=True, persistent_workers=True, num_workers=4) #shuffle=True, 
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=BS, pin_memory=True, persistent_workers=True, num_workers=4)

    return train_loader, test_loader, classes

# From: https://github.com/pytorch/examples/blob/main/mnist/main.py & 
# https://github.com/daisukelab/sound-clf-pytorch/blob/master/Training-Classifier.ipynb

class Net(nn.Module):
    def __init__(self, n_classes):
        super(Net, self).__init__()
        # self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(30, 60),
            nn.ReLU(),
            nn.Linear(60, 60),
            nn.ReLU(),
            nn.Linear(60, n_classes)
        )

    def forward(self, x):
        x = self.linear_relu_stack(x.float())
        x = nn.Softmax(dim=0)(x) # or func.log_softmax
        return x

def training_step(model, device, optimiser, train_loader, num_epochs, log_every_n = 20): #, batch_size
    for epoch in range(num_epochs):
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            y_hat = model(x)

            loss = func.cross_entropy(y_hat, y)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
        if epoch % log_every_n == 0:
            print('Epoch: ', epoch, 'train_loss: ', loss)
        elif epoch == num_epochs-1:
            print('Final epoch: ', epoch, 'train_loss: ', loss)
    return loss

def eval_acc(model, device, dataloader, debug_name=None):
    model = model.to(device).eval()
    count = correct = 0
    for X, gt in dataloader:
        outputs = model(X.to(device))
        _, preds = torch.max(outputs, dim=1)
        correct += sum(preds == gt.to(device))
        count += len(gt)
    acc = correct/count
    if debug_name:
        print(f'{debug_name} acc = {acc:.4f}')
    return acc


def main():
    train_loader, test_loader, classes = pytorchPackageData(GetNewestDataFileName())

    if torch.cuda.is_available():
        device = torch.device('cuda:0' if torch.cuda.get_device_name('cuda:0') == 'NVIDIA GeForce GTX 1070' else 'cuda:1')
    else:
        device = torch.device('cpu')

    print(torch.cuda.get_device_name(device=device))
    
    n_classes = len(classes)
    neural_model = Net(n_classes).to(device)
    print(neural_model)

    optimiser = torch.optim.SGD(neural_model.parameters(), lr=5e-4, momentum=0.95)

    training_step(neural_model, device, optimiser, train_loader, num_epochs=900)

    eval_acc(neural_model, device, test_loader, 'test')

if __name__ == "__main__":
    main()