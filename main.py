import pickle

import tensorflow as tf
import numpy as np
import os
import cv2
from tqdm import tqdm

CATEGORIES = ("cats", "dogs")
RES_X = 50
RES_Y = 50

CATS_DIR = os.path.join(os.getcwd(), "cats")
DOGS_DIR = os.path.join(os.getcwd(), "dogs")

X = []
Y = []


def loadImageCategory(path, category):
    filenames = os.listdir(path)
    for filename in filenames:
        X.append(category)
        Y.append(filename)


def build_output(index):
    out = [0 for _ in CATEGORIES]
    out[index] = 1
    return out


for i, cat in enumerate(CATEGORIES):
    DIR = os.path.join(os.getcwd(), f"images\\{cat}")

    for img_name in tqdm(os.listdir(DIR)):
        try:
            gray_img = cv2.imread(os.path.join(DIR, img_name), cv2.IMREAD_GRAYSCALE)
            resized = cv2.resize(gray_img, (RES_X, RES_Y))
            # plt.imshow(resized, cmap="gray");
            # plt.show()

            X.append(resized)
            Y.append(build_output(i))
        except:
            print("King")

X = np.array(X).reshape((-1, RES_X, RES_Y, 1))
print(X)
Y = np.array(Y)

X = X / 255

pickle.dump(X, open("pickles/X.pickle", "wb"), protocol=4)
pickle.dump(Y, open("pickles/Y.pickle", "wb"), protocol=4)
