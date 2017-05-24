from keras.layers import Input, Embedding, Merge, Dense, Activation, Flatten
from keras.layers.merge import multiply, Dot, concatenate, dot
from keras.models import Sequential, Model
from keras import metrics
from utils import *

all_questions, all_sentences, f1_scores = get_data()
words = get_words(all_sentences)

# dictionaries for converting words to integers and vice versa
word2idx = dict((v, i) for i, v in enumerate(words))
idx2word = list(words)

# convert sentences into a numpy array
sentences_idx = [to_idx(sentence, word2idx=word2idx) 
                            for sentence in all_sentences]
sentences_array = np.asarray(sentences_idx, dtype='float32')
# convert questions into a numpy array
questions_idx = [to_idx(question, word2idx=word2idx)
                            for question in all_questions]
questions_array = np.asarray(questions_idx, dtype='float32')



# parameters for the model
sentence_maxlen = 200
question_maxlen = 200    #longest question has 31 words though...
n_words = len(set(words)) # n_words = 84547
N_EMBED_DIMS = 200
batch_size = 128


# sentence branch
input_sentence = Input(shape=(sentence_maxlen,), dtype='float32')
input_emb_sent = Embedding(n_words, N_EMBED_DIMS)(input_sentence)

# flatten for later concatenation
flattened_input_emb_sent = Flatten()(input_emb_sent)

# question branch
input_question = Input(shape=(question_maxlen,), dtype='float32')
input_emb_quest = Embedding(n_words, N_EMBED_DIMS)(input_question)

########################################################################
# TODO:
# make layer multiply trainable, i.e. multply with additional trainable weights.
# For that implement own layer.
########################################################################

# computing similarity
similarity = multiply([input_emb_sent, input_emb_quest])
similarity = Flatten()(similarity)


# concatenate sentence and similarity result
concatenated = concatenate([flattened_input_emb_sent, similarity], axis=-1)
activated = Activation('relu')(concatenated)

# final fully connected layer
d = Dense(1)(activated)


model = Model(inputs=[input_sentence, input_question], outputs=[d])
model.compile(optimizer='sgd', 
                loss='mean_squared_error', 
                metric=['mae', 'acc'])



model.fit([sentences_array, questions_array], [f1_scores], epochs=1, verbose=1)
            #batch_size=batch_size, verbose=1)




















