from subprocess import call
import os

def main():
    wd = os.getcwd()
    print(wd)
    os.chdir(wd)
    #call(["python", "P7-EIT-AAU\\LabelClipsToCSV.py"])
    call(["python", "P7-EIT-AAU\\ComapareMLToAudacity.py"])
if __name__ == "__main__":
    main()