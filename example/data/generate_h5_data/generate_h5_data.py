
import json, os
import h5py
import numpy as np
from util import *

MAX_WORDS = 25 # the maximum length of sentence in train and test is 24
setting = [('train', True), ('test-dev', True), ('test-standard', True), ('train', False), ('test-dev', False), ('test-standard', False)]
batch_stream_length = 100000
BUFFER_SIZE = 50

DATA_DIR = './data/'

OUTPUT_DIR = DATA_DIR + 'h5_data/buffer_%d' % BUFFER_SIZE
OUTPUT_DIR_PATTERN = '%s/%%s_batches' % OUTPUT_DIR

if os.path.exists(OUTPUT_DIR):
    import shutil
    shutil.rmtree(OUTPUT_DIR)
os.makedirs(OUTPUT_DIR)

# load train data
f_train_ans = open(DATA_DIR + 'Annotations/mscoco_train2014_annotations.json')
train_ans = json.load(f_train_ans)
train_answer = ans_preprocessing(train_ans['annotations'])
answer_idx = [e for e,d in enumerate(train_answer)]
train_answer = [train_answer[d] for d in answer_idx]
train_image_id = np.array(['train2014/COCO_train2014_000000' + str(train_ans['annotations'][d]['image_id']).zfill(6) for d in answer_idx])

f_train = open(DATA_DIR + 'Questions/OpenEnded_mscoco_train2014_questions.json')
train_data = json.load(f_train)
train_question = ques_preprocessing(train_data['questions'])
train_question_id = [d['question_id'] for d in train_data['questions']]

# Load validation data
f_val_ans = open(DATA_DIR + 'Annotations/mscoco_val2014_annotations.json')
val_ans = json.load(f_val_ans)
val_answer = ans_preprocessing(val_ans['annotations'])
answer_idx = [e for e,d in enumerate(val_answer)]
val_answer = [val_answer[d] for d in answer_idx]
val_image_id = np.array(['val2014/COCO_val2014_000000' + str(val_ans['annotations'][d]['image_id']).zfill(6) for d in answer_idx])

f_val = open(DATA_DIR + 'Questions/OpenEnded_mscoco_val2014_questions.json')
val_data = json.load(f_val)
val_question = ques_preprocessing(val_data['questions'])
val_question_id = [d['question_id'] for d in val_data['questions']]

print len(train_question), len(val_question)
print len(train_answer), len(val_answer)
print len(train_image_id), len(val_image_id)

# combine train_answer and val_answer to get the answer vocabulary and question vocabulary
train_val_answer = train_answer + val_answer
train_val_image_id = np.concatenate((train_image_id, val_image_id), axis=0)

# generate answer vocab
answer_out_path = '%s/answer_vocabulary.txt' % OUTPUT_DIR
answer_vocab,answer_inverted = init_answer_vocabulary(train_val_answer, min_count=23)   ### 24:990 answers || 23:1032 answers
dump_vocabulary(answer_inverted, answer_out_path)

# only keep the train question whose answer is in top500                                                                   
answer_idx_top500 = [e for e,d in enumerate(train_val_answer) if d[0] in answer_inverted]
train_val_answer = [train_val_answer[d] for d in answer_idx_top500]
train_val_image_id = np.array([train_val_image_id[d] for d in answer_idx_top500])

train_val_question = train_question + val_question
train_val_question_id = train_question_id + val_question_id
train_val_question = [train_val_question[d] for d in answer_idx_top500]  # only keep quesiton whose answer in top500 answers  
train_val_question_id = [train_val_question_id[d] for d in answer_idx_top500]  # only keep quesiton_id whose answer in top500

# generate question vocab
question_vocab_out_path = '%s/question_vocabulary.txt' % OUTPUT_DIR
question_vocab,question_vocab_inverted = init_vocabulary(train_val_question, min_count=3)  # min_count=2:words:9441 # min_count=3:words:7477
dump_vocabulary(question_vocab_inverted, question_vocab_out_path)


# Load test-dev data
f_testD = open(DATA_DIR + 'Questions/OpenEnded_mscoco_test-dev2015_questions.json')
testD_data = json.load(f_testD)
testD_question = ques_preprocessing(testD_data['questions'])
testD_question_id = [d['question_id'] for d in testD_data['questions']]
testD_image_id = [d['image_id'] for d in testD_data['questions']]

# Load test-standard data
f_testS = open(DATA_DIR + 'Questions/OpenEnded_mscoco_test2015_questions.json')
testS_data = json.load(f_testS)
testS_question = ques_preprocessing(testS_data['questions'])
testS_question_id = [d['question_id'] for d in testS_data['questions']]
testS_image_id = [d['image_id'] for d in testS_data['questions']]


# Combine Dataset
# train 
train_dataset = []
for i in xrange(len(train_val_question)):
  temp = {}
  temp['answer']=train_val_answer[i]
  temp['image_id']=[train_val_image_id[i]]
  temp['question']=train_val_question[i]
  temp['question_id']=train_val_question_id[i]
  train_dataset.append(temp)

# test-dev
testD_dataset = []
for i in xrange(len(testD_question)):
  temp = {}
  temp['question']=testD_question[i]
  temp['question_id']=testD_question_id[i]
  temp['image_id']=[testD_image_id[i]]
  testD_dataset.append(temp)

# test-standard
testS_dataset = []
for i in xrange(len(testS_question)):
  temp = {}
  temp['question']=testS_question[i]
  temp['question_id']=testS_question_id[i]
  temp['image_id']=[testS_image_id[i]]
  testS_dataset.append(temp)

## start to generate hdf data
for (split_name,aligned) in setting:
  if split_name == 'train':
     annotations = train_dataset[0:]
  elif split_name == 'test-dev':
     annotations = testD_dataset[0:]
  elif split_name == 'test-standard':
     annotations = testS_dataset[0:]

  # after alignment, the data is shuffled  
  output_dataset_name = split_name
  if aligned:
    annotations = align_dataset(annotations, BUFFER_SIZE)
    output_dataset_name += '_aligned_%d' % MAX_WORDS
  else:
    annotations = shuffle_dataset(annotations)
    output_dataset_name += '_unaligned_%d'% MAX_WORDS
  output_path = OUTPUT_DIR_PATTERN % output_dataset_name
  os.makedirs(output_path)
  
  num = len(annotations)
  num_batches = (num-1) / batch_stream_length + 1  # a total of three batches
  files = []
  image_out_path = '%s/image_list.txt' % output_path
  images_tmp=list()
  images_tmp_GT_ans = list()
  for i in xrange(num_batches):
    batch = annotations[i*batch_stream_length:(i+1)*batch_stream_length:]
    if split_name == 'train':
        streams = get_streams(batch, question_vocab, answer_vocab, MAX_WORDS)
    else:
        streams = get_streams_test(batch, question_vocab, MAX_WORDS)
    filename = '%s/batch_%d.h5' % (output_path, i)
    print len(batch)

    question = np.zeros([ len(batch), MAX_WORDS ])
    question_id = np.zeros(len(batch))
  
    for idx in xrange(len(batch)):
      stream = streams[idx]
      row = idx
      question[row,]   = np.array(stream['question'])
      question_id[idx] = np.array(batch[idx]['question_id'])

    for idy in xrange(len(streams)):
      if split_name=='train':
          name = 'train'
          images_tmp.append('%s%s.jpg %d' % (DATA_DIR, batch[idy]['image_id'][0], streams[idy]['answer'][0]))
      elif split_name=='test-dev':
          name = 'test'
          images_tmp.append('%s%s2015/COCO_%s2015_000000%s.jpg %d' % (DATA_DIR,name,name,str(batch[idy]['image_id'][0]).zfill(6),0))
      elif split_name=='test-standard':
          name = 'test'
          images_tmp.append('%s%s2015/COCO_%s2015_000000%s.jpg %d' % (DATA_DIR,name,name,str(batch[idy]['image_id'][0]).zfill(6),0))

    files.append(filename)
    h5file = h5py.File(filename, 'w')
    dataset = h5file.create_dataset('question', shape=question.shape, dtype=question.dtype)
    dataset[:] = question
    dataset = h5file.create_dataset('question_id', shape=question_id.shape, dtype=question_id.dtype)
    dataset[:] = question_id
    h5file.close()

  dump_image_file(image_out_path, images_tmp)
  filelist = '%s/hdf5_chunk_list.txt' % output_path
  print 'dumping hdf data: ',filelist
  print '\n'
  with open(filelist, 'wb') as listfile:
    for f in files:
      listfile.write('%s\n' % f)
  
