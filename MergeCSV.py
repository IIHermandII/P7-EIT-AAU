import os 
import pandas as pd

def main():
    envP7RootDir = os.getenv("P7RootDir")
    if envP7RootDir is None:
        print("---> If you are working in vscode\n---> you need to restart the aplication\n---> After you have made a env\n---> for vscode to see it!!")
        print("---> You need to make a env called 'P7RootDir' containing the path to P7 root dir")
        raise ValueError('Environment variable not found (!env)')
    dirPath = envP7RootDir + "\\Data\\CSV files self"

    CSVlist = []
    # Load the two datasets
    print(os.listdir(dirPath))
    for filename in os.listdir(dirPath):
        if filename.endswith(".csv"):  # Only process WAV files
            CSVlist.append(pd.read_csv(dirPath+"\\"+filename))

    # Concatenate the datasets
    merged_df = pd.concat(CSVlist, ignore_index=True)

    # Save the merged dataset to a new CSV file
    outputPath = dirPath + "\\Merged_training_dataset.csv"
    merged_df.to_csv(outputPath, index=False)

if __name__ == "__main__":
    main()