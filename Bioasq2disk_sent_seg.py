import json
from urllib.request import urlopen
import xml.etree.ElementTree as ET
import csv
import pickle
from nltk import sent_tokenize


batch_size = 100

def url2abstract(url):
    # print("opening "+url+"...")
    xmldata = urlopen(url).read().decode('utf-8')
    # clean up the data
    xmldata = xmldata.replace("&gt;", ">")
    xmldata = xmldata.replace("&lt;", "<")
    xmldata = xmldata.replace("\n", "")
    # create xml tree
    root = ET.fromstring(xmldata)
    # return text of abstract, even if Abstract contains subsections
    result = ""
    sentences = []

    for abst in root.findall(".//AbstractText"):
        result=result + str(abst.text) + " "
    # print("abstract was extracted from url sucessfully")
    return result

def write_picklefile(dataset, filename):
    """
    Saves data on disk as a pickle file. This might not be the best option 
    since it may depend on the machine/python version for writing and loading
    """
    with open(filename, 'wb') as picklefile:
        pickle.dump(dataset, picklefile, protocol=pickle.HIGHEST_PROTOCOL)


with open('BioASQ-trainingDataset5b.json') as json_data:
    dataset = json.load(json_data)
    print("Json data imported!")
    json_data.close()


# for debugging take just 5 questions
#dataset = dataset["questions"][:5]
dataset = dataset["questions"]

# new dataset will look like:
# [query, question, [abstracts]]
new_dataset = []
nb_skipped = 0

for i in range(len(dataset)):
    try:
        print("Extracting abstracts from question ", i)
        abstracts=[]
        question = dataset[i]["body"]
        summary=dataset[i]["ideal_answer"]

        # extract abstracts as a list of abstracts
        for url in dataset[i]["documents"]:
            abstracts.append(url2abstract(url+"?report=xml"))
        ##############################################################
        # NOTE: summary is sometimes a list with repeated sentences  #
        # this will be fixed later (segment%identify.py) in order to #
        # avoid downloading the data again                           #
        ##############################################################
        new_dataset.append([question, summary, abstracts])

        # write data in batches
        if (i%batch_size==0 and i!=0):
            write_picklefile(new_dataset[i-batch_size:i], 
                "batch_"+str(i-batch_size)+"to"+str(i))
        elif (len(dataset)-(i%batch_size) < batch_size) and (i!=0):
            write_picklefile(new_dataset[i-batch_size:i], 
                "batch_"+str(len(dataset)-str(i-batch_size))+"to"+str(i))
    except:
        print("Question "+str(i)+" skipped")
        nb_skipped += 1

# write whole dataset in disk if no questions were skipped
with open('batch_all', 'wb') as picklefile:
    pickle.dump(new_dataset, picklefile, protocol=pickle.HIGHEST_PROTOCOL)

    # save data on disk as csv
    with open('task5b.csv', 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter="$", 
            quoting=csv.QUOTE_MINIMAL)
        writer.writerows(new_dataset)



# when loading the file use following code:
#with open('task5b.pickle', 'wb') as picklefile:
#    new_dataset_imported = pickle.load(picklefile)













