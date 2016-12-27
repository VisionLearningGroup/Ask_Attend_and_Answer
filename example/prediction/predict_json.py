
import sys,os,json
import numpy as np
caffe_root = '../caffe/'
sys.path.insert(0, caffe_root + 'python')
import caffe

data_root = './data/h5_data/buffer_50/'
work_root = './prediction/'
sys.path.insert(0, work_root)
from util import *

iter = 243000

IMAGE_FOLDER = './data/test2015/'
DEPLOY       = work_root + 'deploy.prototxt'
MODEL        = work_root + 'mm_iter_%s.caffemodel' % iter
RESULT       = work_root + 'test/result_%s.json' % iter

VOCABULARY        = data_root + 'question_vocabulary.txt'
VOCABULARY_ANSWER = data_root + 'answer_vocabulary.txt'
VAL_INFO          = data_root + 'test-dev_unaligned_25_batches/batch_0.h5'
VAL_INFO_list     = data_root + 'test-dev_unaligned_25_batches/hdf5_chunk_list.txt'    
VAL_LIST          = data_root + 'test-dev_unaligned_25_batches/image_list.txt'

caffe.set_mode_gpu()
caffe.set_device(0)
lstmnet = caffe.Net(DEPLOY, MODEL, caffe.TEST)

val_image, val_hashid = load_image_label(VAL_LIST)  
val_h5 = load_hdf5(VAL_INFO)     
coco_voc = np.array(load_voc(VOCABULARY))   
coco_voc_A = np.array(load_voc(VOCABULARY_ANSWER))   

size = val_h5['question_id'].shape[0]
batch_size = (size - 1) / 25 + 1
json_data = []
bar_length = 20
for batch in xrange(batch_size):
   batch_question = val_h5['question'][batch*25:(batch+1)*25]
   batch_qid = val_h5['question_id'][batch*25:(batch+1)*25]
   res = lstmnet.forward()
   batch_pred = res['prob'].argmax(axis=1)
   hashes = '#' * int(round(batch*25.0/size * bar_length))
   spaces = ' ' * (bar_length - len(hashes))
   sys.stdout.write("\r[%s]%.2f%%  " % (hashes+spaces,batch*25.0/size*100))
   sys.stdout.flush()
   for idx, sentence in enumerate(batch_question):
      tmp_data = {}
      tmp_data['answer'] = coco_voc_A[batch_pred[idx]]  # prediction answer
      tmp_data['question_id'] = batch_qid[idx]  # question id
      json_data.append(tmp_data)

sys.stdout.write("\n")
json.dump(json_data,open(RESULT,'w')) 
