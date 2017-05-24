"""
This script produces labels of all sentences in the dataset by computing
the F1 ROUGE score between a sentence and a target summary belonging 
to the same question. 
"""

import pickle
from nltk import sent_tokenize

from rouge import Rouge




###########################
# segmenting dataset
###########################

with open('batch_all', 'rb') as picklefile:
    dataset=pickle.load(picklefile)

# Segment document into passages (i.e sentences):
# dataset = [question, summary, [abstract1, abstract2,...], 
#            [[abst1_sent1, abst1_sent2, ...], [a2s1, a2s2,...],...]]
new_dataset = []

for i in range(len(dataset)):
    sentences = []
    question = dataset[i][0]
    summary = dataset[i][1]
    abstracts = dataset[i][2]
    for abst in abstracts:
        # split each abstract into sentences
        sentences.append(sent_tokenize(abst))
    new_dataset.append([question, summary, abstracts, sentences])

print("data segmented!")
# write dataset to disk
with open('segmented_data', 'wb') as picklefile:
    pickle.dump(new_dataset, picklefile, protocol=pickle.HIGHEST_PROTOCOL)
    print("segmented data written to disk!")


with open('segmented_data', 'rb') as picklefile:
    segmented_data = pickle.load(picklefile)
print("segmented data has been read from disk!")

###########################
# Computing labels
###########################

rouge = Rouge()

labeled_dataset = []
for elm in segmented_data:
    question = elm[0]
    summary = elm[1]
    # make sure summary is a string
    if type(summary)==type([]):
        summary = summary[0]
    docs = elm[3]
    labels = []
    
    # get a list of sentences per set of documents
    docsset_as_sentences = []
    for doc in docs:
        for sent in doc:
            docsset_as_sentences += [sent]
   
    # compute F1 Rouge score for each sentence
    for sentence in docsset_as_sentences:
        f1_score = rouge.get_scores(sentence, summary)[0]["rouge-1"]['f']
        labels.append(f1_score)
        
    labeled_dataset.append([docsset_as_sentences, labels, question, summary])



with open("labeled_dataset", "wb") as picklefile:
    pickle.dump(labeled_dataset, picklefile, protocol=pickle.HIGHEST_PROTOCOL)
print("labeled data has been written to disk!")

# produce labeled question dataset
labeled_question_dataset=[]
for elm in labeled_dataset:
    docsset_as_sentences = elm[0]
    labels = elm[1]
    question = elm[2]
    summary = elm[3]
    
    for i in range(len(docsset_as_sentences)):
        labeled_question_dataset.append([question, 
                                        docsset_as_sentences[i], 
                                        labels[i],
                                        summary])

with open("labeled_question_dataset", "wb") as picklefile:
    pickle.dump(labeled_question_dataset, picklefile, 
        protocol=pickle.HIGHEST_PROTOCOL)
print("labeled question dataset has been read from disk!")
print("done!")
        



