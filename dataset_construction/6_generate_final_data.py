import os
import csv
import json
import pandas as pd
from tqdm import tqdm
import hashlib

DATA_DIR = '../data/kg_tmp'
output_directory = '../data/kg_tmp'
text_data_directory = '../data/text_tmp'

final_output_directory = '../data/MIRAI'
if not os.path.exists(final_output_directory):
    os.makedirs(final_output_directory)

if __name__ == "__main__":
    df = pd.read_csv(os.path.join(DATA_DIR, 'kg_source.csv'), sep='\t', dtype=str)
    dict_md52text = json.load(open(os.path.join(text_data_directory, 'dict_md52text_document_filtered.json')))

    # keep text available kg event records as trusted data
    df['URLMD5'] = [hashlib.md5(x.encode()).hexdigest() for x in df['SOURCEURL']]
    df_wtext = df[df['URLMD5'].isin(list(dict_md52text.keys()))]
    df_wtext.to_csv(os.path.join(output_directory, 'kg_wtext.csv'), index=False, sep='\t')
    print(f'kg_wtext.csv saved, length: {len(df_wtext)}')

    # build text data files: data_news.csv
    md5s = df_wtext['URLMD5'].unique().tolist()
    dict_md52docid = {}
    for docid, md5 in enumerate(md5s):
        dict_md52docid[md5] = docid

    df_final = df_wtext.copy()
    df_final['Docid'] = [dict_md52docid[x] for x in df_final['URLMD5']]
    df_final.to_csv(os.path.join(final_output_directory, 'data_final.csv'), index=False, sep='\t')
    print(f'data_final.csv saved, length: {len(df_final)}')

    dates = df_final['DateStr'].tolist()
    dict_md52date = {}
    for idx, md5 in enumerate(df_final['URLMD5'].tolist()):
        dict_md52date[md5] = dates[idx]

    dict_docid2text = {}  # {docid: {'MD5', 'URL', 'Date', 'Title', 'Text', 'Abstract'}}

    for idx, (md5, docid) in tqdm(enumerate(dict_md52docid.items()), total=len(dict_md52docid)):
        title = dict_md52text[md5]['Title']
        text = dict_md52text[md5]['Text']
        paragraphs = text.split('\n')
        abstract = title + '\n' + paragraphs[0]
        for par in paragraphs[1:]:
            if len(abstract) > 50:
                break
            abstract += '\n' + par
        dict_docid2text[docid] = {
            'MD5': md5,
            'URL': dict_md52text[md5]['SOURCEURL'],
            'Date': dict_md52date[md5],
            'Title': title,
            'Text': text,
            'Abstract': abstract
        }

    Docids = list(dict_docid2text.keys())
    MD5s = [dict_docid2text[docid]['MD5'] for docid in Docids]
    URLs = [dict_docid2text[docid]['URL'] for docid in Docids]
    Dates = [dict_docid2text[docid]['Date'] for docid in Docids]
    Titles = [dict_docid2text[docid]['Title'] for docid in Docids]
    Texts = [dict_docid2text[docid]['Text'] for docid in Docids]
    Abstracts = [dict_docid2text[docid]['Abstract'] for docid in Docids]

    df_text = pd.DataFrame({'Docid': Docids, 'MD5': MD5s, 'URL': URLs, 'Date': Dates, 'Title': Titles, 'Text': Texts, 'Abstract': Abstracts})
    df_text.to_csv(os.path.join(final_output_directory, 'data_news.csv'), index=False, sep='\t')
    print(f'data_news.csv saved, length: {len(df_text)}')

    # build kg data: df_kg.csv (dedup by (s, r, o, t, d) from data_final.csv)
    dict_quadcode2docids = {}
    quadcodes = df_final['QuadEventCode'].tolist()
    docids = df_final['Docid'].tolist()
    for idx, quadcode in enumerate(quadcodes):
        if quadcode not in dict_quadcode2docids:
            dict_quadcode2docids[quadcode] = []
        dict_quadcode2docids[quadcode].append(docids[idx])

    df_final['Docids'] = [dict_quadcode2docids[x] for x in df_final['QuadEventCode']]
    df_kg = df_final[['DateStr', 'Actor1CountryCode', 'Actor2CountryCode', 'EventBaseCode',
                      'Actor1CountryName', 'Actor2CountryName', 'RelName',
                      'QuadEventCode', 'QuadEventName',
                      'Docid', 'Docids']]
    df_kg = df_kg.drop_duplicates(subset=['QuadEventCode', 'Docid'], ignore_index=True)

    df_kg.to_csv(os.path.join(final_output_directory, 'data_kg.csv'), index=False, sep='\t')
    print(f'data_kg.csv saved, length: {len(df_kg)}')