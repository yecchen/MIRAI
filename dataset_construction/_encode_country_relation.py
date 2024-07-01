import openai
import os
import json
import numpy as np
from tqdm import tqdm

OPENAI_API_KEY = os.environ['OPENAI_API_KEY']

def get_embedding(text, model="text-embedding-3-small"):
   text = text.replace("\n", " ")
   return openai.Embedding.create(input = [text], model=model).data[0].embedding

# country
dict_iso2alternames = json.load(open('../data/info/dict_iso2alternames_GeoNames.json', 'r'))

# get embeddings for the first country name of each iso
country_embeddings = []
for idx, iso in tqdm(enumerate(dict_iso2alternames), total=len(dict_iso2alternames)):
    country_name = dict_iso2alternames[iso][0]
    country_embeddings.append(get_embedding(country_name))
country_embeddings = np.array(country_embeddings)
print(country_embeddings.shape)
# save embeddings
np.save('../data/info/country_embeddings.npy', country_embeddings)

# relation
dict_cameo2relation = json.load(open('../data/info/dict_code2relation.json', 'r'))

# get embeddings for a concatenation of the relation name and its description
relation_embeddings = []
for idx, cameo in tqdm(enumerate(dict_cameo2relation), total=len(dict_cameo2relation)):
    relation_name = dict_cameo2relation[cameo]['Name']
    relation_description = dict_cameo2relation[cameo]['Description']
    relation_embeddings.append(get_embedding(relation_name + ': ' + relation_description))
relation_embeddings = np.array(relation_embeddings)
print(relation_embeddings.shape)
# save embeddings
np.save('../data/info/relation_embeddings.npy', relation_embeddings)

