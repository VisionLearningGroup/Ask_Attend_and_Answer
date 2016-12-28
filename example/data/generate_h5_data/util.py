import random
import numpy as np
random.seed(3)
#random.seed(time.clock())

UNK_IDENTIFIER = '<unk>'
def init_vocabulary(image_annotations, min_count=1):
  words_to_count = {}
  for annotation in image_annotations:
    for word in annotation:
      word = word.strip()
      if word not in words_to_count:
        words_to_count[word] = 0
      words_to_count[word] += 1
  # Sort words by count, then alphabetically
  words_by_count = sorted(words_to_count.keys(), key=lambda w: (-words_to_count[w], w))
  print 'Initialized vocabulary with %d words; top 10 words:' % len(words_by_count)
  for word in words_by_count[:10]:
    print '\t%s (%d)' % (word, words_to_count[word])
  # Add words to vocabulary
  vocabulary = {UNK_IDENTIFIER: 0}
  vocabulary_inverted = [UNK_IDENTIFIER]
  for index, word in enumerate(words_by_count):
    word = word.strip()
    if words_to_count[word] < min_count:
      break
    vocabulary_inverted.append(word)
    vocabulary[word] = index + 1
  print 'Final vocabulary (restricted to words with counts of %d+) has %d words' % \
      (min_count, len(vocabulary))
  return vocabulary, vocabulary_inverted

###### init the vocabulary of answers  #######
def init_answer_vocabulary(image_annotations, min_count=1):
  words_to_count = {}
  for annotation in image_annotations:
    for word in annotation:
      word = word.strip()
      if word not in words_to_count:
        words_to_count[word] = 0
      words_to_count[word] += 1
  # Sort words by count, then alphabetically
  words_by_count = sorted(words_to_count.keys(), key=lambda w: (-words_to_count[w], w))
  print 'Initialized vocabulary with %d words; top 10 words:' % len(words_by_count)
  for word in words_by_count[:10]:
    print '\t%s (%d)' % (word, words_to_count[word])
  # Add words to vocabulary
  #  vocabulary = {UNK_IDENTIFIER: 0}
  #  vocabulary_inverted = [UNK_IDENTIFIER]
  vocabulary = {}
  vocabulary_inverted = []
  for index, word in enumerate(words_by_count):
    word = word.strip()
    if words_to_count[word] < min_count:
      break
    vocabulary_inverted.append(word)
    #    vocabulary[word] = index + 1      # add one for the <UNK> token
    vocabulary[word] = index           # add one for the <UNK> token
  vocabulary_inverted.append(UNK_IDENTIFIER)
  #  vocabulary[UNK_IDENTIFIER] = 1001    # I set this number 1001, so ignore 1001 when training
  vocabulary[UNK_IDENTIFIER] = 2001 
  print 'Final vocabulary (restricted to words with counts of %d+) has %d words' % \
      (min_count, len(vocabulary))
  return vocabulary, vocabulary_inverted



def dump_vocabulary(vocabulary_inverted, vocab_filename):
  print 'Dumping vocabulary to file: %s' % vocab_filename
  with open(vocab_filename, 'w') as vocab_file:
    for word in vocabulary_inverted:
      vocab_file.write('%s\n' % word)
  print 'Done.'

def align_dataset(annotations, batch_num_streams, shuffle=True):
  num_pairs = len(annotations)
  remainder = num_pairs % batch_num_streams
  if remainder > 0:
    num_needed = batch_num_streams - remainder
    for i in range(num_needed):
      choice = random.randint(0, num_pairs - 1)
      annotations.append(annotations[choice])
  assert len(annotations) % batch_num_streams == 0
  if shuffle:
    random.shuffle(annotations)
  return annotations

def shuffle_dataset(annotations, shuffle=True):
  if shuffle:
    random.shuffle(annotations)
  return annotations

def line_to_stream(sentence, vocabulary):
  stream = []
  for word in sentence:
    word = word.strip()
    if word in vocabulary:
      stream.append(vocabulary[word])
    else:  # unknown word; append UNK
      stream.append(vocabulary[UNK_IDENTIFIER])
  # increment the stream -- 0 will be the EOS character
  # stream = [s + 1 for s in stream]
  # stream = [s for s in stream]   # no need for EOS token in memory network
  return stream

def get_streams(annotations, vocab, vocab_answer, max_words):
  streams = []
  for annotation in annotations:
    #    image    = annotation['image_id']
    question = annotation['question']
    answer   = annotation['answer']
    question = line_to_stream(question, vocab)
    answer = line_to_stream(answer, vocab_answer)
    out = {}
    out['question'] = question + [-1] * (max_words - len(question))
    out['answer'] = answer
    streams.append(out)
  return streams

def get_streams_test(annotations, vocab, max_words):
  streams = []
  for annotation in annotations:
    question = annotation['question']
    question = line_to_stream(question, vocab)
    out = {}
    out['question'] = question + [-1] * (max_words - len(question))
    streams.append(out)
  return streams

def dump_image_file(image_filename, image_list, dummy_image_filename=None):
  print 'Dumping image list to file: %s' % image_filename
  with open(image_filename, 'wb') as image_file:
    for image_path in image_list:
      image_file.write('%s\n' % image_path)
  if dummy_image_filename is not None:
    print 'Dumping image list with dummy labels to file: %s' % dummy_image_filename
    with open(dummy_image_filename, 'wb') as image_file:
      for path_and_hash in image_list:
        image_file.write('%s %d\n' % path_and_hash)
  print 'Done.'

def ques_preprocessing(data):
   return [d['question'].replace('?',' ?').replace(',',' , ').replace('&',' ').replace('.','').replace('/','').replace('"',' ').replace('#','').replace('<','').replace('>','').replace('(','').replace(')','').lower().split() for d in data]

def ans_preprocessing(data):
   return [[d['multiple_choice_answer'].replace(',',' ').replace('&',' ').replace('/','').replace('"',' ').replace('#','').replace(';',' ').replace('<','').replace('>','').replace('(','').replace(')','').lower()] for d in data]




