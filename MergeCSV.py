import os 
import pandas as pd
import re


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
    dirDates = sorted(dirDates,key=lambda l:l[1],reverse=True) # Take oldest data first i belive 
    return(workDir + "\\" + dirDates[0][1])


def main():
    envP7RootDir = os.getenv("P7RootDir")
    if envP7RootDir is None:
        print("---> If you are working in vscode\n---> you need to restart the aplication\n---> After you have made a env\n---> for vscode to see it!!")
        print("---> You need to make a env called 'P7RootDir' containing the path to P7 root dir")
        raise ValueError('Environment variable not found (!env)')
    dirPath = envP7RootDir + "\\Data\\CSV files self"

    CSVlist = []
    # Load the datasets (self)
    print(os.listdir(dirPath))
    for filename in os.listdir(dirPath):
        if filename.endswith(".csv"):  # Only process WAV files
            CSVlist.append(pd.read_csv(dirPath+"\\"+filename))

    # Concatenate the datasets
    merged_df = pd.concat(CSVlist, ignore_index=True)

    #Load our dataset
    NewestDataFileName = GetNewestDataFileName() 
    our_df = pd.read_csv(NewestDataFileName)

    #Add the two datasets (our+self labelled)
    total_df = pd.concat([our_df, merged_df], ignore_index=True)
    outputPath = envP7RootDir + "\\Data\\Total data file.csv"
    total_df.to_csv(outputPath, index=False)

if __name__ == "__main__":
    main()