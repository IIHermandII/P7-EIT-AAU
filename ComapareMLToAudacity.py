import os
from sklearn.preprocessing import LabelEncoder, StandardScaler  # Import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split, cross_val_score  # Import cross_val_score
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tqdm import tqdm
import soundfile as sf
import sys
from termcolor import colored

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
    for i, lines in enumerate(f):
        Data = lines.split()
        Data[0] = float(Data[0])
        if Data[0] > 170: # 201 because it seams seen in audasaty that the system fucks up there
            break 
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

        if i > 200: # 201 because it seams seen in audasaty that the system fucks up there
            break   # acc 0.52 --> 0.68

        lst = []
        #print("Test nr: " + str(i) + " Data: " + str(TestData[i][0])+ " " + str(TestData[i][1]) + " " + str(TestData[i][2]))
        for j in range(len(PedictData)):
            # start point is with in test data , # end point is with in test data, # test data is with in predected data, # test data is bigger that prediction
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
    list = ['BI','BO','CT','M','V']
    cm_disp = ConfusionMatrixDisplay(conf_matrix, display_labels=list)
    #Plot Confusion matrices
    cm_disp.plot(cmap = "Blues")
    plt.title('Confusion Matrix: Actual vs Predicted')

    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=["BI","BO","CT","M","V"], yticklabels=["BI","BO","CT","M","V"])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()     

def GetFileInformation(PedictData, TestData):
    CorrectTimeArrayBI = []
    CorrectTimeArrayBO = []
    CorrectTimeArrayV = []
    CorrectTimeArrayCT = []
    CorrectTimeArrayM = []
    RongTimeArray = []
    CorrectCheckSum = [] 

    def GetFileInFormationMath(StartTime, EndTime, PedictData, TestData):
        TotalTime = EndTime - StartTime
        CorrectCheckSum.append(TotalTime)
        #print("Total time: ", TotalTime)
        #print(f"TotalTime: {TotalTime}, Start: {StartTime}, End: {EndTime}, Prediction: {PedictData}, Test: {TestData}")
        match TestData:
            case "BI":
                if PedictData == "BI":
                    CorrectTimeArrayBI.append(TotalTime)
                else:
                    RongTimeArray.append([[TotalTime],["TD : ",TestData, " PD : ", PedictData]])
            case "BO":
                if PedictData == "BO":
                    CorrectTimeArrayBO.append(TotalTime)
                else:
                    RongTimeArray.append([[TotalTime],["TD : ",TestData, " PD : ", PedictData]])
            case "V":
                if PedictData == "V":
                    CorrectTimeArrayV.append(TotalTime)
                else:
                    RongTimeArray.append([[TotalTime],["TD : ",TestData, " PD : ", PedictData]])
            case "CT":
                if PedictData == "CT":
                    CorrectTimeArrayCT.append(TotalTime)
                else:
                    RongTimeArray.append([[TotalTime],["TD : ",TestData, " PD : ", PedictData]])
            case "M":
                if PedictData == "M":
                    CorrectTimeArrayM.append(TotalTime)
                else:
                    RongTimeArray.append([[TotalTime],["TD : ",TestData, " PD : ", PedictData]])
    def ResultPresentation(note,ArrayBI,ArrayBO,ArrayV,ArrayCT,ArrayM,CheckSum):
        print("----------------------------")
        print("Results of Labeling ", note, ": \nPrediction resulted in the folowing times:\n")
        print("BI: ", sum(ArrayBI))
        print("BO: ", sum(ArrayBO))
        print("V: ", sum(ArrayV))
        print("CT: ", sum(ArrayCT))
        print("M: ", sum(ArrayM))
        print("Total Sum: ", sum([sum(ArrayBI),sum(ArrayBO),sum(ArrayV),sum(ArrayCT),sum(ArrayM)]))
        print("Check Sum: ", CheckSum)
        print("----------------------------")
                
    DataCase = [PedictData, TestData]
    for index, DC in enumerate(DataCase,start=1):
        TimeArrayBI = [] # resets arrays 
        TimeArrayBO = []
        TimeArrayV = []
        TimeArrayCT = []
        TimeArrayM = []
        ChechSum = 0
        for i in range(len(DC)):
            #if index == 2:
                #print(f"TotalTime: {DC[i][1]-DC[i][0]}, Start: {DC[i][0]}, End: {DC[i][1]}")
            ChechSum += (DC[i][1]-DC[i][0])
            match DC[i][2]:
                case "BI":
                    TimeArrayBI.append(DC[i][1]-DC[i][0])
                case "BO":
                    TimeArrayBO.append(DC[i][1]-DC[i][0])
                case "V":
                    TimeArrayV.append(DC[i][1]-DC[i][0])
                case "CT":
                    TimeArrayCT.append(DC[i][1]-DC[i][0])
                case "M":
                    TimeArrayM.append(DC[i][1]-DC[i][0])
        
        if index == 1:
            strInfo = "System prediction"
        else:
            strInfo = "System we have labeled"
        print(colored(strInfo, 'green', attrs=['bold']))
        ResultPresentation(strInfo,TimeArrayBI,TimeArrayBO,TimeArrayV,TimeArrayCT,TimeArrayM,ChechSum)

    for i in range(len(TestData)):
        #print("ends times" ,TestData[i][1])
        for j in range(len(PedictData)):
            #print("ends times" ,PedictData[j][1])
                # Start in test data 
            if (PedictData[j][0]>=TestData[i][0] and PedictData[j][0]<=TestData[i][1]):
                if (PedictData[j][1]<=TestData[i][1]):
                    # print(1)
                    # Predictet data is with in Test data
                    #                       StartTime,      EndTime,        PedictDataName,     TestDataName
                    GetFileInFormationMath(PedictData[j][0],PedictData[j][1],PedictData[j][2],TestData[i][2])
                else:
                    # print(2)
                    # Predictet data starts but dosent end in Test data
                    #                       StartTime,      EndTime,        PedictDataName,     TestDataName
                    GetFileInFormationMath(PedictData[j][0],TestData[i][1],PedictData[j][2],TestData[i][2])
                # Ends in test data
            elif (PedictData[j][1]>=TestData[i][0] and PedictData[j][1]<=TestData[i][1]):
                # print(3)
                #                       StartTime,      EndTime,        PedictDataName,     TestDataName
                GetFileInFormationMath(TestData[i][0],PedictData[j][1],PedictData[j][2],TestData[i][2])

                # Test Data is with in 
            elif (PedictData[j][0]<=TestData[i][0] and PedictData[j][1]>=TestData[i][1]):
                # print(4)
                #                       StartTime,      EndTime,        PedictDataName,     TestDataName
                GetFileInFormationMath(TestData[i][0],TestData[i][1],PedictData[j][2],TestData[i][2])
    print(colored("Prediction macth with our labeling", 'green', attrs=['bold']))
    ResultPresentation("correct Time",CorrectTimeArrayBI,CorrectTimeArrayBO,CorrectTimeArrayV,CorrectTimeArrayCT,CorrectTimeArrayM,"1")
    print("Check Sum: ", sum(CorrectCheckSum))
    summ = 0
    for i in RongTimeArray:
        # print((i[0][0]))
        summ += i[0][0]
    print(summ)
    print("BI : ", f"{sum(CorrectTimeArrayBI)/sum(TimeArrayBI)*100:.2f}", "%\tBO : ", f"{sum(CorrectTimeArrayBO)/sum(TimeArrayBO)*100:.2f}", "%\tV : ", f"{sum(CorrectTimeArrayV)/sum(TimeArrayV)*100:.2f}", "%\tCT : ", f"{sum(CorrectTimeArrayCT)/sum(TimeArrayCT)*100:.2f}", "%\t M : ", f"{sum(CorrectTimeArrayM)/sum(TimeArrayM)*100:.2f}","%")
  
def main():
    RootDir = sys.argv[1]
    PathToPredectetData = RootDir + "\\Outputs\\Predictions_2_Smart(SVM).txt"
    PathToTestData = RootDir + "\\SoundFile\\LablingWeHaveDone.txt"
    PedictData = loadDataToList(PathToPredectetData)
    TestData = loadDataToList(PathToTestData)
    GetFileInformation(PedictData, TestData)    
    # CompareLists(PedictData, TestData)
    # ConfusionMatrix()
    # print(LstBI)
    
if __name__ == "__main__":
    main()