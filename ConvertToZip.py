import shutil, os

print(os.getcwd())

def convertToZip(path):
    shutil.make_archive(path, 'zip', path)
    print("Zip file created")
    shutil.rmtree(path)
    print("Original directory deleted")
    pass