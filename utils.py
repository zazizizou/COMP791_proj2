import pickle
from nltk.tokenize import RegexpTokenizer
import numpy as np

from keras import backend as k
from keras.engine.topology import Layer


# Print iterations progress
def printProgressBar (iteration, 
                        total, 
                        prefix = '', 
                        suffix = '', 
                        decimals = 1, 
                        length = 100, 
                        fill = '█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals 
                                        in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    # Print New Line on Complete
    if iteration == total: 
        print()




def get_data():
    """
    return questions, all_sentences and their respective f1_scores
    """
    with open('labeled_question_dataset', 'rb') as picklefile:
        labeled_question_dataset = pickle.load(picklefile)

    all_questions = []
    all_sentences = []
    f1_scores = []
    all_summaries = []
    for i in range(len(labeled_question_dataset)):
        all_questions += [labeled_question_dataset[i][0]]
        all_sentences += [labeled_question_dataset[i][1]]
        f1_scores += [labeled_question_dataset[i][2]]
        all_summaries += [labeled_question_dataset[i][3]]

    f1_scores = np.asarray(f1_scores, dtype='float32')

    return all_questions, all_sentences, f1_scores, all_summaries

def get_words(all_sentences):
    """
    returns the vocabulary, given a list of sentences
    all_sentences: list of strings.
    """
    punct_tokenizer = RegexpTokenizer(r'\w+')
    all_words_ever = []
    for i in range(len(all_sentences)):
        all_words_ever += punct_tokenizer.tokenize(all_sentences[i])
    words = [word.lower() for word in all_words_ever]
    words = set(all_words_ever)
    return words


def to_idx(sentence, word2idx, sentence_maxlen, zeropad=True, no_index=0):
    """
    converts a sentence into an array of word indices using
    the dictionary word2idx.
    no_index is the index of a word if it is not found in the 
    dictionary word2idx. 
    """
    punct_tokenizer = RegexpTokenizer(r'\w+')
    words = punct_tokenizer.tokenize(sentence)
    indexes = []
    for word in words:
        if word in word2idx.keys():
            indexes.append(word2idx[word])
        else:
            indexes.append(no_index)
    
    # zero pad
    if zeropad == True:
        while len(indexes) <= sentence_maxlen:
            indexes.append(0)
    # truncate
    return indexes[:sentence_maxlen]





def get_embedding_matrix(words, 
                        word2idx,
                        N_EMBED_DIMS, 
                        types_path="word2vecTools/types.txt",
                        vector_path="word2vecTools/vectors.txt"):
    """
    Compute embedding matrix of shape (n_words, N_EMBED_DIMS)
    @params:
        words: vocabulary (list or set).
        word2idx: function transforming words to indixes (python function).
        N_EMBED_DIMS: size of word embedding vector (integer)
        types_path: path to file containing words (string)
        vector_path: path to file containing word embeddings of words in 
                    types_path (string)
    """
    print("Computing embedding matrix...")

    words = list(words)
    embedding_index = {}


    

    print("getting embedding indexes...")
    types=[]
    types_idx=[]
    embedding_index={}

    with open(types_path, "r") as types_file:
        for idx, line in enumerate(types_file):
            word = line.replace("\n", "")
            types.append(word)
            if word in words:
                types_idx.append(idx)
        
    with open(vector_path, "r") as vector_file:
        for i, line in enumerate(vector_file):
            if i in types_idx:
                vector = line.replace("\n", "").split()
                embedding_index[types[i]]= np.asarray(vector, dtype='float32')
                        

    print("converting to embedding matrix...")                    
    embedding_matrix = np.zeros((len(words), N_EMBED_DIMS))
    for word, i in word2idx.items():
        try:
            embedding_matrix[i] = embedding_index[word.lower()]
        except KeyError:
            print(" Following word was not found in vocabulary:", word)

    print("embedding matrix shape:", embedding_matrix.shape)

    # free a bit of memory
    del embedding_index

    # return final result
    return embedding_matrix



def to_idxarray(all_sentences, word2idx, sentence_maxlen):
    """
    converts a sentence into a numpy array of indexes
    """
    sentences_idx = [to_idx(sentence, word2idx=word2idx, 
                            sentence_maxlen=sentence_maxlen) 
                                for sentence in all_sentences]
    return np.asarray(sentences_idx, dtype='float32')



class Similarity(Layer):
    """
    This custom layer performs a weighted point wise multiplication.
    The weights are trainable.  
    """

    def __init__(self, **kwargs):
        super(Similarity, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(shape=(input_shape[1], self.output_dim),
                                    initializer='uniform',
                                    trainable=True)
        super(Similarity, self).build(input_shape)

    def call(self, input1, inpu2):
        x = k.multiply(self.kernel, input1)
        return k.multiply(x, inputs2)

    def compute_output_shape(self, input_shape):
        shape1 = list(input_shape[0])
        shape2 = list(input_shape[1])
        return (shape1)







