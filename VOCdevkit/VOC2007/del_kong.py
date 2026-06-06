import numpy as np
import os

path = "./labels"
photo = "./JPEGImages"
for i in os.listdir(path):
    a = np.loadtxt(os.path.join(path,i))
    print("dasd")
    if len(a)==0:
        print(i)
        os.remove(os.path.join(path,i))
        ab = i.split('.')[0]
        ab = ab+".jpg"
        os.remove(os.path.join(photo, ab))