import os
from sklearn.preprocessing import LabelEncoder, StandardScaler  # Import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split, cross_val_score  # Import cross_val_score
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

LstBI = []
LstBO = []
LstV = []
LstCT = []
LstM = []


def loadDataToList(pathToData):
    print("Working dir: ")
    print(os.getcwd())
    f = open(pathToData, "r")
    DataList = []
    for lines in f:
        Data = lines.split()
        Data[0] = float(Data[0])
        Data[1] = float(Data[1])
        DataList.append([Data[0],Data[1],Data[2]])
    return DataList

def PutAnswerinToList(PedictData, TestData):
        # BI, BO, V, CT, M
    match TestData:
        case "BI":
            LstBI.append(PedictData)
        case "BO":
            LstBO.append(PedictData)
        case "V":
            LstV.append(PedictData)
        case "CT":
            LstCT.append(PedictData)
        case "M":
            LstM.append(PedictData)

def CompareLists(PedictData, TestData):
    for i in range(len(TestData)):
        lst = []
        #print("Test nr: " + str(i) + " Data: " + str(TestData[i][0])+ " " + str(TestData[i][1]) + " " + str(TestData[i][2]))
        for j in range(len(PedictData)):
            # start point is with in test data , # end point is with in test data, # test data is with in predected data
            if (PedictData[j][0]>TestData[i][0] and PedictData[j][0]<TestData[i][1]) or (PedictData[j][1]>TestData[i][0] and PedictData[j][1]<TestData[i][1]) or (PedictData[j][0]<TestData[i][0] and PedictData[j][1]>TestData[i][1]):
                lst.append(PedictData[j][2])
                PutAnswerinToList(PedictData[j][2], TestData[i][2])
  
def ConfusionMatrix():
    y_pred = []
    y_pred.extend(LstBI)
    y_pred.extend(LstBO)
    y_pred.extend(LstV)
    y_pred.extend(LstCT)
    y_pred.extend(LstM)
    #print(y_pred)
    y_test_encoded = []
    Target = [LstBI, LstBO, LstV, LstCT, LstM]
    for i in range(len(Target)):
        for j in range(len(Target[i])):
            match i:
                case 0:
                    y_test_encoded.append("BI")
                case 1:
                    y_test_encoded.append("BO")
                case 2:
                    y_test_encoded.append("V")
                case 3:
                    y_test_encoded.append("CT")
                case 4:
                    y_test_encoded.append("M")
    #print(y_test_encoded)
    # Calculate accuracy
    print("BI :" + str(len(LstBI)))
    print("BO :" + str(len(LstBO)))
    print("V  :" + str(len(LstV)))
    print("CT :" + str(len(LstCT)))
    print("M  :" + str(len(LstM)))

    accuracy = accuracy_score(y_test_encoded, y_pred)
    print(f'Accuracy: {accuracy:.2f}')

    # Print classification report
    print(classification_report(y_test_encoded, y_pred, target_names=["BI","BO","CT","M","V"])) # Just how it is 

    # Confusion Matrix Visualization
    conf_matrix = confusion_matrix(y_test_encoded, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=["BI","BO","CT","M","V"], yticklabels=["BI","BO","CT","M","V"])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix: Actual vs Predicted')
    plt.show()     

def main():
    PathToPredectetData = "P7-EIT-AAU\\Outputs\\Predictions 2 (SVM).txt"
    PathToTestData = "P7-EIT-AAU\\Outputs\\TestLabels 2.txt"
    PedictData = loadDataToList(PathToPredectetData)
    TestData = loadDataToList(PathToTestData)
    CompareLists(PedictData, TestData)
    ConfusionMatrix()
    print(LstBI)
    
if __name__ == "__main__":
    main()