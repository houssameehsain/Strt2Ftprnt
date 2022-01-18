import os
import re
import cv2
from tqdm import tqdm

""" 
    Get the corresponding ground truth images 
    for the input street network images
"""

# raw data directories
X_img_path = "D:/AI in Urban Design/DL UD dir/data_X"
Y_img_path = "D:/AI in Urban Design/DL UD dir/data_Y"

wanted = []
for i in tqdm(sorted(os.listdir(Y_img_path)), ncols=100, disable=False):
    path = os.path.join(Y_img_path, i)
    wanted.append(int(re.search(r'\d+', i).group()))

for j in tqdm(sorted(os.listdir(X_img_path)), ncols=100, disable=False):
    if int(re.search(r'\d+', j).group()) in wanted:
        path = os.path.join(X_img_path, j)
        im = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        cv2.imwrite('D:/AI in Urban Design/DL UD dir/data_X/img_X_'
                + str(int(re.search(r'\d+', j).group())) + '.png', im)
    else:
        continue