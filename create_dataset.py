# create_dataset.py
# Creates a dataset from a directory of images

import gzip, cPickle
import os
import cv2
import numpy as np
def get_data():
    Y = []
    X = []
    for dirpath, subdirs, files in os.walk('./data'):
        dirname = dirpath.split(os.path.sep)[-1]
        if dirname == 'data':
            num = len(subdirs)
        if dirname != 'data':
            label = np.zeros(num) 
            label[int(dirname)] = 1.0
        for x in files:
            path = os.path.join(dirpath, x)
            img = cv2.imread(path).reshape(3, 224, 224)
            X.append(img)
            Y.append(label)
    X = np.asarray(X)
    train_X = X[:int(0.7*len(X))]
    test_X = X[int(0.7*len(X)):]
    Y = np.asarray(Y)
    train_Y = X[:int(0.7*len(Y))]
    test_Y = X[int(0.7*len(Y)):]
    dataset = [train_X, test_X, train_Y, test_Y]
    
    f = gzip.open('dataset.pkl.gz', 'wb')
    cPickle.dump(dataset, f, protocol=2)
    f.close
    

def get_data_deprecated():
    neutral_path = '/home/halwa/code/projects/minor/dataset/neutral/'
    attacker_path = '/home/halwa/code/projects/minor/dataset/attack/'
    defender_path = '/home/halwa/code/projects/minor/dataset/defend/'
    train_data = []
    test_data = []
    train_labels = []
    test_labels= []
    for i in range(1, 41):
        neutral_img_path = neutral_path + str(i) + '.jpg'
        img = cv2.imread(neutral_img_path)
 #       img.reshape(3, 224, 224)
        if i < 20:
            train_data.append(img)
            train_labels.append([0, 0, 1])
        else:
            test_data.append(img)
            test_labels.append([0, 0, 1])
    for i in range(1, 5):
        defender_img_path = defender_path + str(i) + '.png'
        img = cv2.imread(defender_img_path)
        #img.reshape(3, 224, 224)
        if i < 3:
            train_data.append(img)
            train_labels.append([0, 1, 0])
        else:
            test_data.append(img)
            test_labels.append([0, 1, 0])
    for i in range(1, 6):
        attacker_img_path = attacker_path + str(i) + '.png'
        img = cv2.imread(attacker_img_path)
        #img.reshape(3, 224, 224)
        if i < 4:
            train_data.append(img)
            train_labels.append([1, 0, 0])
        else:
            test_data.append(img)
            test_labels.append([1, 0, 0])
    return np.array(train_data).reshape(24, 3,224,224), np.array(test_data).reshape(25, 3,224,224), np.array(train_labels).reshape(24, 3), np.array(test_labels).reshape(25, 3)
