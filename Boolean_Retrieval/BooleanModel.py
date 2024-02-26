import os
import json
import numpy as np
import re
from logical_operation import *
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

        # Logical operations
        self.operators = {
            '&': AND,
            '|': OR,
            '~': NOT
        }

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
    
    def str_query(self, query):
        tokens = self.__preprocess(query)
        inverted_values = []
        for key in tokens:
            if key in self.inverted_index:
                inverted_values.append(set(self.inverted_index[key]))

        try:
            intersection = set.intersection(*inverted_values)
            # docs = [self.documents[idx] for idx in intersection]
            return intersection
        except:
            return []
        
    def __onehot_encoding(self, index_list):
        if index_list:
            binary_list = [1 if i in index_list else 0 for i in range(len(self.documents))]
        else:
            binary_list = [0] * len(self.documents)
        return binary_list

    def logic_query(self, text):
        regex = r'\(+.+?\)+|[\w]+|\S+'
        result = re.findall(regex, text)

        if len(result) % 2 == 1:        
            value = result[0]
            
            if value[0] == '(' and value[0] == ')':
                fn_str = self.logic_query(result[0])
            else:
                index_list = self.inverted_index.get(value)
                fn_str = self.__onehot_encoding(index_list)

            for i in range(1, len(result), 2):
                value = result[i+1]
                operation = result[i]
                
                if value[0] == '(' and value[-1] == ')':
                    value = value[1:-1]
                    encoded_value = self.logic_query(value)
                    fn_str = self.operators[operation]([fn_str, encoded_value])
                else:
                    if value[0] == '~':
                        index_list = self.inverted_index.get(value[0])
                        operation = value[0]
                        encoded_value = self.__onehot_encoding(index_list)
                        neg_encoded_value = self.operators[operation](encoded_value)
                        fn_str = self.operators[operation]([fn_str, neg_encoded_value])
                    else:
                        index_list = self.inverted_index.get(value)
                        encoded_value = self.__onehot_encoding(index_list)
                        fn_str = self.operators[operation]([fn_str, encoded_value])
            
            indices = [index for index, value in enumerate(fn_str) if value]                    
            return indices