import os
import shutil
import zipfile

'''This script will unzip the file and will put all the videos in the LE2I folder that is created. The names of the videos correspond to the file labels.csv'''

path = 'path where the FallDataset.zip is' # download the dataset from https://imvia.u-bourgogne.fr/en/database/fall-detection-dataset-2.html
zipfil = 'FallDataset.zip'

path_un  = os.path.join(path,zipfil)

with zipfile.ZipFile(path_un, 'r') as zip_ref:
    zip_ref.extractall(path)


folders = ['Coffee_room_01', 'Coffee_room_02', 'Home_01', 'Home_02', 'Lecture_room', 'Office']

for folder in folders:
    
    # os.system('mkdir '+ os.path.join(path,folder))
    
    with zipfile.ZipFile(os.path.join(path,folder)+'.zip', 'r') as zip_ref:
        zip_ref.extractall(path)

os.rename(os.path.join(path,'Lecture room'), os.path.join(path,'Lecture_room'))
   
os.mkdir(os.path.join(path, 'dataset')) 
     
for folder in folders:
    
    files = os.listdir(os.path.join(path, folder))
    
    if len(files) == 2:
        files2 = os.listdir(os.path.join(path, folder, 'Videos'))
        for f in files2:
            src = os.path.join(path, folder, 'Videos', f) 
            dst = os.path.join(path, 'dataset', folder+'_'+f) 
            
            shutil.move(src, dst)
    else:
        for f in files:
            src = os.path.join(path, folder, f) 
            dst = os.path.join(path, 'dataset', folder+'_'+f) 
            
            shutil.move(src, dst)
    

