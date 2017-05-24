import pickle
import numpy as np


types_path="word2vecTools/types.txt"
vector_path="word2vecTools/vectors.txt"


vectors = []
embedding_index = {}

with open(vector_path, "r") as vector_file:
    for line in vector_file:
        vectors.append(np.asarray(line.replace("\n", "").split()))

with open(types_path, "r") as types_file:
    for i, line in enumerate(types_file):
        word = line.replace("\n", "")
        embedding_index[word] = vectors[i]



# save embedding indexes
with open("embedding_index", "w") as picklefile:
    pickle.dump(embedding_index, picklefile, protocol=pickle.HIGHEST_PROTOCOL)