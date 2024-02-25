import os
import json
import numpy as np
from tqdm.auto import tqdm
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from collections import defaultdict


class BooleanModel:
    def __init__(self, documents, save_data_path, over_write=False):
        self.documents = documents
        
        # Get stopwords
        self.STOPWORDS = set(stopwords.words("english"))

        # For stemming purposes
        self.STEMMER = PorterStemmer()

        # Create vocabularies
        vocab_path = os.path.join(save_data_path, 'vocab.npy')
        if os.path.isfile(vocab_path) and over_write==False:
            self.vocab_list = np.load(vocab_path, allow_pickle=True)
        else:
            self.vocab_list = self.__create_vocabulary(documents, vocab_path)

        # Create inverted index
        inverted_index_path = os.path.join(save_data_path, 'inverted_index.json')
        if os.path.isfile(inverted_index_path) and over_write==False:
            self.inverted_index = json.loads(open(inverted_index_path, "r").read())
        else:
            self.inverted_index = self.__create_inverted_index(documents, inverted_index_path)


    def __create_vocabulary(self, documents, save_path):
        print('\nCreate vocabulary')
        process_bar = tqdm(range(len(documents)))
        vocab_list = set()
        for doc in documents:
            tokens = self.__preprocess(doc)
            vocab_list.update(tokens)
            process_bar.update(1)
        vocab_list = list(vocab_list)

        # Save list
        np.save(save_path, vocab_list, allow_pickle=True)
        return vocab_list

    def __preprocess(self, doc):
        tokens = word_tokenize(doc)
        tokens = [self.STEMMER.stem(token.lower()) for token in tokens if token not in self.STOPWORDS]
        return tokens

    def __create_inverted_index(self, documents, save_path=None):
        print('\nCreate inverted index')
        inverted_index = defaultdict(list)
        process_bar = tqdm(range(len(self.vocab_list)))
        for vocab in self.vocab_list:
            for id, doc in enumerate(documents):
                tokens = doc.split(" ")
                if vocab in tokens:
                    inverted_index[vocab].append(id)
            process_bar.update(1)

        if save_path!=None:
            with open(save_path, "w") as json_file:
                json.dump(inverted_index, json_file)

        return inverted_index
    
    def query(self, query):
        tokens = self.__preprocess(query)
        inverted_values = []
        for key in tokens:
            if key in self.inverted_index:
                inverted_values.append(set(self.inverted_index[key]))

        try:
            intersection = set.intersection(*inverted_values)
            docs = [self.documents[idx] for idx in intersection]
            return docs
        except:
            return []