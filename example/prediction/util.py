import numpy as np
import h5py

def load_voc(filename):
    f = open(filename)
    data = f.readlines()
    f.close()
    return [d[:-1] for d in data]

def load_hdf5(file_name):
    h5data  = h5py.File(file_name, 'r')
    question = np.array([list(c) for c in h5data['question']])
    question_id = h5data['question_id']
    val_h5={}
    val_h5['question'] = question
    val_h5['question_id'] = question_id
    return val_h5

def load_image_label(file_name):
    data = np.loadtxt(file_name, str, delimiter=' ')
    image = data[:,0]
    image = np.array([im.split('/')[-1] for im in data[:,0]])
    hashid = data[:,1]
    return image, hashid


