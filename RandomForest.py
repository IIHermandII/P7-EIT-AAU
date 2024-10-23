import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import joblib
import os
import re
import seaborn as sns
from sklearn.decomposition import PCA
import numpy as np

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
    NewestDataFileName = GetNewestDataFileName() 
    # Load the merged dataset
    df = pd.read_csv(NewestDataFileName)

    #Remove filename coloum
    X = df.drop(['Filename'], axis=1)

    #Keep data and lables
    data = X.iloc[:, 1:].values
    labels = X.iloc[:, 0].values

    print("Data length: ", len(data))

    data_S = StandardScaler().fit_transform(data)

    # Split the dataset into training and testing sets (80/20)
    x_train, x_test, y_train, y_test = train_test_split(data_S, labels, test_size=0.2, random_state=420)

    lda = LDA(n_components=4).fit(x_train,y_train)
    x_train_LDA = lda.transform(x_train)
    x_test_LDA = lda.transform(x_test)

    print("LDA explained var: ",sum(lda.explained_variance_ratio_))

    # Initialize the Random Forest classifier
    RF = RandomForestClassifier(n_estimators=100, random_state=42)
    GBDT = GradientBoostingClassifier(n_estimators=100, random_state=42)

    # Train the model on the training set
    RF.fit(x_train, y_train)
    GBDT.fit(x_train, y_train)

    # Predict on the testing set
    y_pred = RF.predict(x_test)
    #y_pred = GBDT.predict(x_test)

    # Evaluate the model
    print(f"Accuracy on Test Set: {accuracy_score(y_test, y_pred):.4f}")
    print("Classification Report on Test Set:")
    print(classification_report(y_test, y_pred))

    # Visualize feature importance
    feature_importances = RF.feature_importances_
    print(feature_importances)
    # Feature names are the columns in X_train
    feature_names = df.drop(['Filename', 'Label'], axis=1).columns

    # Create a pandas Series to view the feature importances with their corresponding feature names
    importances = pd.Series(feature_importances, index=feature_names)

    # Sort the features by importance
    sorted_importances = importances.sort_values(ascending=False)

    plt.figure(figsize=(10, 8))
    sns.barplot(x=sorted_importances, y=sorted_importances.index)
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title('Feature Importance')

    # Save the model to disk
    #filename = 'finalized_model20.sav'
    #joblib.dump(model, filename)

    # Load the model from disk
    #loaded_model = joblib.load(filename)
    
    #Confusion matrix
    
    #Compute the confusion matrices for PCA and LDA
    pca_cm = confusion_matrix(y_test, y_pred,normalize='true')

    #Prepare for plotting
    list = ['BI','BO','CT','M','V']
    cm = ConfusionMatrixDisplay(pca_cm, display_labels=list)

    #Plot Confusion matrices
    cm.plot(cmap = "Blues")

    pca = PCA()
    principalComponents = pca.fit_transform(data_S)

    # Create a biplot
    def biplot(score, coeff, labels=None):
        xs = score[:,0]
        ys = score[:,1]
        scalex = 1.0/(xs.max() - xs.min())
        scaley = 1.0/(ys.max() - ys.min())
        plt.scatter(xs * scalex, ys * scaley, c='gray') # Plotting the scaled scores
        for i in range(len(coeff)):
            # Plot arrows and labels for each variable
            x, y = coeff[i,0], coeff[i,1]
            plt.arrow(0, 0, x, y, color='red', alpha=0.5)
            if labels is None:
                plt.text(x* 1.15, y * 1.15, f"Var{i+1}", color='green', ha='center', va='center')
            else:
                plt.text(x* 1.15, y * 1.15, labels[i], color='green', ha='center', va='center')
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.grid()

    # Create the figure for the biplot
    plt.figure(figsize=(12, 8))
    biplot(principalComponents[:, :2], np.transpose(pca.components_[0:2, :]), labels=feature_names)
    plt.title('Biplot of first two Principal Components')

    plt.show()

if __name__ == "__main__":
    main()