from keras.layers import Input, Embedding, Merge, Dense
from keras.layers import Activation, Flatten, Lambda
from keras.layers.merge import multiply, Dot, concatenate, dot
from keras.models import Sequential, Model
from keras import metrics
from utils import *
from math import floor, ceil
from copy import deepcopy


###################################
# define functions for Lambda layer
###################################
def get_mean(x):
    from keras import backend as k
    return k.mean(x, axis=1, keepdims=True)
def get_mean_output_shape(input_shape):
    return (input_shape[0], input_shape[2])



##########################
# parameters for the model
##########################

sentence_maxlen = 195
question_maxlen = 195    #longest question has 31 words though...
N_EMBED_DIMS = 200
batch_size = 128


##########################
# dataset
##########################
# import all dataset
all_questions, all_sentences, f1_scores, all_summaries = get_data()

# for debugging
#all_questions = all_questions[:3]
#all_sentences = all_sentences[:3]
#f1_scores = f1_scores[:3]


words = get_words(all_sentences)
n_words = len(set(words)) # n_words = 84547

# dictionaries for converting words to integers and vice versa
word2idx = dict((v, i) for i, v in enumerate(words))
idx2word = list(words)

# convert sentences into a numpy array
sentences_idx = [to_idx(sentence, word2idx=word2idx, 
                        sentence_maxlen=sentence_maxlen) 
                            for sentence in all_sentences]
sentences_array = np.asarray(sentences_idx, dtype='float32')
# convert questions into a numpy array
questions_idx = [to_idx(question, word2idx=word2idx,
                        sentence_maxlen=question_maxlen)
                            for question in all_questions]
questions_array = np.asarray(questions_idx, dtype='float32')



# import weights for embedding matrix
# embedding_matrix = get_embedding_matrix(words=words, 
#                                        word2idx=word2idx, 
#                                        N_EMBED_DIMS=N_EMBED_DIMS)
embedding_matrix = np.load("embedding_matrix.npy")


##########################
# Regression Model
##########################

# sentence branch
input_sentence = Input(shape=(sentence_maxlen,), dtype='float32')
input_emb_sent = Embedding(n_words, 
                            N_EMBED_DIMS,
                            weights=[embedding_matrix],
                            trainable=False)(input_sentence)
#input_emb_sent = Lambda(lambda x: k.mean(x, axis=1))(input_emb_sent)
input_emb_sent = Lambda(get_mean, get_mean_output_shape)(input_emb_sent)


# question branch
input_question = Input(shape=(question_maxlen,), dtype='float32')
input_emb_quest = Embedding(n_words, 
                            N_EMBED_DIMS,
                            weights=[embedding_matrix],
                            trainable=False)(input_question)
#input_emb_quest = Lambda(lambda x: k.mean(x, axis=1))(input_emb_quest)
input_emb_quest = Lambda(get_mean, get_mean_output_shape)(input_emb_quest)


# computing similarity
similarity = multiply([input_emb_sent, input_emb_quest])


# concatenate sentence and similarity result
concatenated = concatenate([input_emb_sent, similarity], axis=1)
d = Dense(100)(concatenated)
d = Activation('relu')(d)

# final fully connected layer
f1_prediction = Dense(1)(d)


model = Model(inputs=[input_sentence, input_question], outputs=[f1_prediction])
model.compile(optimizer='sgd', 
                loss='mean_squared_error', 
                metric=['mae', 'acc'])



#############################
# train / cross validation
#############################

history = []
nb_folds = 10
nb_epochs = 1

for i in range(nb_folds):
    model.compile(optimizer='sgd', 
                loss='mean_squared_error', 
                metric=['mae', 'acc'])
    print("i=",i)
    i = i/nb_folds
    val_begin = floor(i*len(all_sentences))
    val_end = ceil((i+0.1)*len(all_sentences))

    # partition dataset into training and testing
    X_train_quest = deepcopy(all_questions)
    del X_train_quest[val_begin:val_end]
    X_train_quest = to_idxarray(X_train_quest, 
                                word2idx=word2idx, 
                                sentence_maxlen=question_maxlen)
    X_train_sent = deepcopy(all_sentences)
    del X_train_sent[val_begin:val_end]
    X_train_sent = to_idxarray(X_train_sent , 
                                word2idx=word2idx, 
                                sentence_maxlen=sentence_maxlen)

    X_val_sent = to_idxarray(all_sentences[val_begin:val_end], 
                                word2idx=word2idx, 
                                sentence_maxlen=sentence_maxlen)
    X_val_quest = to_idxarray(all_questions[val_begin:val_end], 
                                word2idx=word2idx, 
                                sentence_maxlen=question_maxlen)
    y_train = list(f1_scores)
    del y_train[val_begin:val_end]
    y_train = np.asarray(y_train, dtype="float32")

    y_val = f1_scores[val_begin:val_end]

    H = model.fit([X_train_quest, X_train_sent], [y_train], 
                epochs=nb_epochs,# verbose=1)
                batch_size=batch_size,
                validation_data=([X_val_quest, X_val_sent], y_val),
                verbose=1)
    history.append(H)



model.save("nnr.h5")
















