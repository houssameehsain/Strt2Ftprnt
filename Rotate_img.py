import os
import cv2
import re
from tqdm import tqdm

# raw data directories
X_img_path = "D:/AI in Urban Design/DL UD dir/STRT2FTPRNT/data_X"
Y_img_path = "D:/AI in Urban Design/DL UD dir/STRT2FTPRNT/data_Y"

for i in tqdm(sorted(os.listdir(X_img_path)), ncols=100, disable=False):
    path = os.path.join(X_img_path, i)
    img = cv2.imread(path)
    rot1_img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    rot2_img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    rot3_img = cv2.rotate(rot1_img, cv2.ROTATE_90_CLOCKWISE)
    cv2.imwrite('D:/AI in Urban Design/DL UD dir/STRT2FTPRNT/data_X/img_X_'
                + str(int(re.search(r'\d+', i).group())) + 'rot1' + '.png', rot1_img)
    cv2.imwrite('D:/AI in Urban Design/DL UD dir/STRT2FTPRNT/data_X/img_X_'
                + str(int(re.search(r'\d+', i).group())) + 'rot2' + '.png', rot2_img)
    cv2.imwrite('D:/AI in Urban Design/DL UD dir/STRT2FTPRNT/data_X/img_X_'
                + str(int(re.search(r'\d+', i).group())) + 'rot3' + '.png', rot3_img)

for j in tqdm(sorted(os.listdir(Y_img_path)), ncols=100, disable=False):
    path = os.path.join(Y_img_path, j)
    img = cv2.imread(path)
    rot1_img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    rot2_img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    rot3_img = cv2.rotate(rot1_img, cv2.ROTATE_90_CLOCKWISE)
    cv2.imwrite('D:/AI in Urban Design/DL UD dir/STRT2FTPRNT/data_Y/img_Y_'
                + str(int(re.search(r'\d+', j).group())) + 'rot1' + '.png', rot1_img)
    cv2.imwrite('D:/AI in Urban Design/DL UD dir/STRT2FTPRNT/data_Y/img_Y_'
                + str(int(re.search(r'\d+', j).group())) + 'rot2' + '.png', rot2_img)
    cv2.imwrite('D:/AI in Urban Design/DL UD dir/STRT2FTPRNT/data_Y/img_Y_'
               + str(int(re.search(r'\d+', j).group())) + 'rot3' + '.png', rot3_img)
