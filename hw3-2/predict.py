import torch
import csv
import pickle
import numpy as np
from model import GAN

PATH = "./model/d_t_1_g_t_3_10_B_N_GAN.model"
PATH_img = "./result/predict"


# cuda
print("enviroment setup")
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

def to_one_hot(array, class_num):
    batch_size = array.shape[0]
    one_hot = np.zeros((batch_size, class_num))
    one_hot[np.arange(batch_size), array] = 1
    one_hot = torch.from_numpy(one_hot).float()
    return one_hot


def predict():
    model = torch.load(PATH)
    model = model.to(device)
    
    with open('./dic/hair.pickle', 'rb') as handle:
        hair_list = pickle.load(handle)
        
    with open('./dic/eye.pickle', 'rb') as handle:
        eye_list = pickle.load(handle)
    
    # get test label
    t_hair_label = []
    t_eye_label = []
    with open("test_label.txt", newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            h = hair_list.index(row[0])
            e = eye_list.index(row[1])
            t_hair_label.append(h)
            t_eye_label.append(e)
    t_hair_label = np.array(t_hair_label)
    t_eye_label = np.array(t_eye_label)
    t_hair_label = to_one_hot(t_hair_label, len(hair_list))
    t_eye_label = to_one_hot(t_eye_label, len(eye_list))
    
    t_hair_label = t_hair_label.to(device)
    t_eye_label = t_eye_label.to(device)
    
    model.predict(PATH_img, t_hair_label, t_eye_label, device)
    
    
    
    
    
    
    
predict()