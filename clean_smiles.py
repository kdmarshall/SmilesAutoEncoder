import os,sys
import pandas as pd
from multiprocessing import Pool

input_file = 'data/version.smi'
output_file = 'data/cleaned.smi'
num_processes = 3
min_chars = 20
max_chars = 100

def remove_salts(smiles):
    if '.' in smiles:
        fragments = smiles.split('.')
        max_length,longest_frag = max([(len(frag),frag) for frag in fragments])
        return longest_frag
    else:
        return smiles

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

def add_uniq_chars(smiles_list):
    uniq_chars = set([])
    for smiles in smiles_list:
        for char in list(smiles):
            if char not in uniq_chars:
                uniq_chars.add(char)
    return uniq_chars

df = pd.read_csv(input_file, delimiter=' ', index_col=None)
df['isosmiles'] = df['isosmiles'].apply(remove_salts)
df = df[df.apply(lambda x: len(x['isosmiles']) >= min_chars and len(x['isosmiles']) <= max_chars, axis=1)]
smiles_list = df['isosmiles'].tolist()
del df['version_id']
del df['parent_id']
df.to_csv('data/cleaned.smi',index=False,header=False)

chunked_list = list(chunks(smiles_list,len(smiles_list)//num_processes))

with Pool(num_processes) as p:
    smiles_sets = p.map(add_uniq_chars, chunked_list)

smiles_chars = sorted(list(smiles_sets[0] | smiles_sets[1] | smiles_sets[2]))
print(smiles_chars)
print(len(smiles_chars))
