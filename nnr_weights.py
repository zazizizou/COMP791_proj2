from keras.layers import Input, Embedding, Merge, Dense
from keras.layers import Activation, Flatten, Lambda
from keras.layers.merge import multiply, Dot, concatenate, dot
from keras.models import Sequential, Model
from keras import metrics
from utils import *


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
all_questions, all_sentences, f1_scores = get_data()

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



# get weights for Embedding layer
embedding_matrix = get_embedding_matrix(words=words, 
                                        word2idx=word2idx, 
                                        N_EMBED_DIMS=N_EMBED_DIMS)

##########################
# Regression Model
##########################

# sentence branch
input_sentence = Input(shape=(sentence_maxlen,), dtype='float32')
input_emb_sent = Embedding(n_words, 
                            N_EMBED_DIMS,
                            weights=[embedding_matrix],
                            trainable=False)(input_sentence)
input_emb_sent = Lambda(lambda x: k.mean(x, axis=1))(input_emb_sent)


# question branch
input_question = Input(shape=(question_maxlen,), dtype='float32')
input_emb_quest = Embedding(n_words, 
                            N_EMBED_DIMS,
                            weights=[embedding_matrix],
                            trainable=False)(input_question)
input_emb_quest = Lambda(lambda x: k.mean(x, axis=1))(input_emb_quest)


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


# train on small dataset
#model.fit([sentences_array[:500], questions_array[:500]], 
#        [f1_scores[:500]], epochs=1, verbose=1)


model.fit([sentences_array, questions_array], [f1_scores], epochs=2,# verbose=1)
            batch_size=batch_size, verbose=1)




















