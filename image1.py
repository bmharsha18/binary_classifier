from PIL import Image
import numpy as np
import os
import csv

temp = os.listdir("/home/harshabm/Documents/Placements/Projects/Neural Network/Untitled Folder/train_64/")
y = list()

for i in range(len(temp)):
    print(i)
    s = temp[i]
    pic = Image.open("/home/harshabm/Documents/Placements/Projects/Neural Network/Untitled Folder/train_64/"+str(s))
    pix = np.array(pic.getdata()).reshape(pic.size[0],pic.size[1],3)
    pix = pix.reshape(12288,1)
    if i==0:
        x = pix
        if 'cat' in s:
            y.append([1])
        else:
            y.append([0])
    if i!=0:
        x = np.append(arr=x,values=pix,axis=1)
        if 'cat' in s:
            y.append([1])
        else:
            y.append([0])       

temp_matrix = x
temp_matrix = temp_matrix / 255
Y = np.array(y)
Y = Y.reshape(1,len(Y))

    
with open("X_train.txt","w") as f:
    writer = csv.writer(f)
    writer.writerows(temp_matrix)
    
with open("Y_train.txt","w") as f:
    writer = csv.writer(f)
    writer.writerows(Y)
    

