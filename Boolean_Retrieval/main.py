import os
import nltk
import pickle
from datasets import load_dataset, load_from_disk
from BooleanModel import *

nltk.download('punkt')
data_path = 'data'

# === Load dataset ===
corpus_path = os.path.join(data_path, 'corpus')
if os.path.exists(corpus_path):
    data = load_from_disk(corpus_path)
else:
    print('Create documents')
    path = 'squad'
    data = load_dataset(path, split="train[:5%]")
    data.save_to_disk(corpus_path)
documents = data['context']
print(f'\nDocuments size: {len(documents)}')


# === Load model === 
model = BooleanModel(documents, save_data_path=data_path, over_write=False)
print(f'\nVocab size: {len(model.vocab_list)}')


# === Infer === 
query = 'explain python language'
print('\nstr_query (Document index):', query)
indices = model.str_query(query)
print(indices)

query = 'explain | test'
print('\nlogic_query (Document index):', query)
indices = model.logic_query(query)
print(indices)

"""# print documents
for indice in indices: 
    print(model.documents[indice])"""


# === Save model ===
with open(os.path.join("saved_model", "model.pkl"), "wb") as f:
    pickle.dump(model, f)
print('\nSaved model successfully')