from PIL import Image
import os

temp = os.listdir("/home/harshabm/Documents/Placements/Projects/Neural Network/animals_64_test")
new_width  = 64
new_height = 64

for i in range(len(temp)):
    s = temp[i]
    img = Image.open('/home/harshabm/Documents/Placements/Projects/Neural Network/animals_64_test/'+str(s))
    img = img.resize((new_width, new_height), Image.ANTIALIAS)
    img.save('/home/harshabm/Documents/Placements/Projects/Neural Network/test/'+str(s))
