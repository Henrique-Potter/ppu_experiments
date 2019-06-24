from pathlib import Path
import cv2 as cv
import os

images = Path('F:\\val2017').glob("*.jpg")

height = 0
width = 0
size = 0

for image in images:
    img_base = cv.imread(str(image))

    height = height + img_base.shape[0]
    width = width + img_base.shape[1]
    size = size + os.stat(str(image)).st_size
    print("Height:{} Width:{} Size:{}".format(height, width, size))

print(int(height/5000))
print(int(width/5000))
print(int(size/5000))
