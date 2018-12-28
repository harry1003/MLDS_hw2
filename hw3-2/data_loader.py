import csv
import numpy as np
import torch
import scipy.misc as misc
root = "./extra_data/"


def get_list(elements):
    e_list = []
    element_set = set(elements)
    for label in element_set:
        e_list.append(label)
    e_list.sort()
    return e_list


def read_label(root):
    file_path = root + "tags.csv"
    hairs = []
    eyes = []
    with open(file_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            hairs.append(row[1].split(' ')[0])
            eyes.append(row[1].split(' ')[2])
  
    return hairs, eyes


class Data_loader():
    def __init__(self, path=root):
        # get label
        hairs, eyes = read_label(root)
        self.hairs = hairs
        self.eyes = eyes
        self.hair_list = get_list(hairs)
        self.eye_list = get_list(eyes)
        
        # def const
        self.data_path = root + "images/"
        self.data_size = len(hairs)
        self.hair_class = len(self.hair_list)
        self.eye_class = len(self.eye_list)
        # def var
        self.idx = np.arange(self.data_size)
        np.random.shuffle(self.idx)
        self.iter = 0

    def load_on_batch(self, batch_size=1, norm=True):
        # reset every epochs
        if self.iter + batch_size > self.data_size:
            np.random.shuffle(self.idx)
            self.iter = 0

        # read_data
        data = []
        hair = []
        eye = []
        for i in range(batch_size):
            num = i + self.iter
            idx = self.idx[num]
            # load img
            img_name = self.data_path + str(idx) + ".jpg"
            img = misc.imread(img_name).transpose(2, 0, 1)
            if norm:
                img = img / 128 - 1
            data.append(img)
            # load label
            hair_c2num = self.hair_list.index(self.hairs[idx])
            eye_c2num = self.eye_list.index(self.eyes[idx])
            
            hair.append(hair_c2num)
            eye.append(eye_c2num)
            
        self.iter = self.iter + batch_size
        data = np.array(data)
        hair = np.array(hair)
        eye = np.array(eye)
        
        # to one hot
        hair2onehot = np.zeros((batch_size, self.hair_class))
        hair2onehot[np.arange(batch_size), hair] = 1
        
        eye2onehot = np.zeros((batch_size, self.eye_class))
        eye2onehot[np.arange(batch_size), eye] = 1

        hair2onehot = torch.from_numpy(hair2onehot).float()
        eye2onehot = torch.from_numpy(eye2onehot).float()
        data = torch.from_numpy(data).float()
        return data, hair2onehot, eye2onehot
    
    def to_one_hot(self, array, class_num):
        batch_size = array.shape[0]
        one_hot = np.zeros((batch_size, class_num))
        one_hot[np.arange(batch_size), array] = 1
        one_hot = torch.from_numpy(one_hot).float()
        return one_hot
        
    def reconstructe(self, img):
        img = (img + 1) * 128
        img = img.transpose(1, 2, 0)
#         misc.imsave("test.jpg", img)
        return img

        

        
        
# d = data_loader()
# d.load_on_batch(10)
# class test():
#     def __init__():
#         self.a = 10
    
#     def p(self):
#         print(self.a)
        
# a = test()
    
    

