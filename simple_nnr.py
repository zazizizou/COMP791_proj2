from keras.layers import Input, Embedding, merge, Flatten, SimpleRNN
from keras.models import Model
import numpy as np
from utils import *


# f1_scores = np.asarray(f1_scores, dtype='float32')
_, all_sentences, f1_scores = get_data()
words = get_words(all_sentences)


# dictionaries for converting words to integers and vice versa
word2idx = dict((v, i) for i, v in enumerate(words))
idx2word = list(words)

# convert sentences into a numpy array
sentences_idx = [to_idx(sentence, word2idx=word2idx) 
                            for sentence in all_sentences]
sentences_array = np.asarray(sentences_idx, dtype='int32')

# parameters for the model
sentence_maxlen = 200
n_words = len(set(words)) # n_words = 84547
N_EMBED_DIMS = 200



# Regression Model
input_sentence = Input(shape=(sentence_maxlen,), dtype='int32')
input_embedding = Embedding(n_words, N_EMBED_DIMS)(input_sentence)
f1_prediction = SimpleRNN(1)(input_embedding)

predict_f1 = Model(input=[input_sentence], output=[f1_prediction])
predict_f1.compile(optimizer='sgd', loss='mean_squared_error')

predict_f1.fit([sentences_array], [f1_scores], nb_epoch = 2, verbose=1)