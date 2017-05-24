from rouge import Rouge
from keras.models import load_model
from utils import *
from copy import deepcopy
import numpy as np

all_questions, all_sentences, _, all_summaries = get_data()

unique_questions = list(set(all_questions))
unique_summaries = list(set(all_summaries))

model = load_model("nnr.h5")
sentence_maxlen = 195
question_maxlen = 195


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



# generate summary
f1_predicted = []
whole_text = []
n = 1000
question = unique_questions[n]
target_summary = unique_summaries[n]
for i in range(len(all_sentences)):
    if all_questions[i] == question:
        sentence = all_sentences[i]
        sentence_array = np.asarray(to_idx(sentence, 
                            word2idx=word2idx, 
                            sentence_maxlen=sentence_maxlen))
        sentence_array = np.reshape(sentence_array, (1, sentence_array.shape[0]))
        question_array = np.asarray(to_idx(question,
                            word2idx=word2idx,
                            sentence_maxlen=question_maxlen))
        question_array = np.reshape(question_array, (1, question_array.shape[0]))
        f1_score = model.predict([question_array, sentence_array])[0][0]
        whole_text.append(sentence)
        f1_predicted.append(f1_score)
        
f1_predicted = np.asarray(f1_predicted, dtype="float32")
whole_text = np.asarray(whole_text)

gen_summary = whole_text[f1_predicted.argsort()] [-2:]
gen_summary = gen_summary[1] + " " + gen_summary[0]

# evaluate summary
rouge = Rouge()
f1_eval = rouge.get_scores(gen_summary, target_summary)[0]["rouge-l"]['f']
print("f1_eval=", f1_eval)

