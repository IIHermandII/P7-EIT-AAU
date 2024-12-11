from subprocess import call
import os
from termcolor import colored

def SoftwareFinnished():
    print(colored("+------------------------------------------------------------------+", 'green', attrs=['bold']))
    print(colored("|This software is made by:\t\t\t\t\t   |", 'green', attrs=['bold']))
    print(colored("|Emil Leth Høimark,\tJens-Ulrik Ladekjær-Mikkelsen\t\t   |", 'green', attrs=['bold']))
    print(colored("|Casper Nøregaard Bak,\tSteffen Damm\t\t\t\t   |", 'green', attrs=['bold']))
    print(colored("|Marcus Mogensen,\tMagnus Høgh\t\t\t\t   |", 'green', attrs=['bold']))
    print(colored("|Aalborg Uneversety [Electrinics engineers master 1. semester 2024]|", 'green', attrs=['bold']))
    print(colored("+------------------------------------------------------------------+", 'green', attrs=['bold']))

def main():
    # The only thing we need you to do is to spicefie the path to the 
    # 'Data' Folder we have providet e.g 
    # PathToDataFolder = C:\\Users\\name\\GIT\\P7-EIT-AAU\\Data
    PathToDataFolder = ""

    print(colored("[1 : 5], LabelClipsToCSV.py", 'green', attrs=['bold']))
    call(["python", "LabelClipsToCSV.py", PathToDataFolder])

    print(colored("[2 : 5], CSVDataVisualizer.py", 'green', attrs=['bold']))
    call(["python", "CSVDataVisualizer.py", PathToDataFolder])

    print(colored("[3 : 5], Pipeline.py", 'green', attrs=['bold']))
    call(["python", "Pipeline.py", PathToDataFolder])

    print(colored("[4 : 5], SmartLableAudio.py", 'green', attrs=['bold']))
    call(["python", "SmartLableAudio.py", PathToDataFolder])

    print(colored("[5 : 5], ComapareMLToAudacity.py", 'green', attrs=['bold']))
    call(["python", "ComapareMLToAudacity.py",PathToDataFolder])

    SoftwareFinnished()
if __name__ == "__main__":
    main()