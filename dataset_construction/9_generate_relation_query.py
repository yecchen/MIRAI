import argparse
import os
import json
import pandas as pd
from datetime import datetime
from tqdm import tqdm

def generate_relation_query(df, full_df, output_path):
    # generate query code (d, s, ?, o)
    dates = df ['DateStr'].tolist()
    actor1 = df['Actor1CountryCode'].tolist()
    actor2 = df['Actor2CountryCode'].tolist()
    query_codes = []
    for i in range(len(dates)):
        date = dates[i]
        a1 = actor1[i]
        a2 = actor2[i]
        query_code = '({}, {}, ?, {})'.format(date, a1, a2)
        query_codes.append(query_code)
    df['QueryCode'] = query_codes

    # generate query to answer relation list
    unique_query_codes = df['QueryCode'].unique().tolist()
    answer_df = full_df[full_df['RelQueryCode'].isin(unique_query_codes)]
    print('answer_df length', len(answer_df))

    dict_query2rel_list = {}
    for idx, query_code in tqdm(enumerate(unique_query_codes), total=len(unique_query_codes)):
        dict_query2rel_list[query_code] = sorted(answer_df[answer_df['RelQueryCode'] == query_code]['EventBaseCode'].unique().tolist())

    # generate query to relation dictionary
    dict_query2rel_dict = {}
    for query, rels in dict_query2rel_list.items():
        dict_query2rel_dict[query] = {}
        for rel in rels:
            if rel[:2] not in dict_query2rel_dict[query]:
                dict_query2rel_dict[query][rel[:2]] = []
            dict_query2rel_dict[query][rel[:2]].append(rel)

    # generate query to answer relations dataframe
    df_query = df.drop_duplicates(subset=['QueryCode'], ignore_index=True)
    unique_query_codes = df_query['QueryCode'].tolist()
    df_query['AnswerList'] = [dict_query2rel_list[query] for query in unique_query_codes]
    df_query['AnswerDict'] = [dict_query2rel_dict[query] for query in unique_query_codes]

    # generate dates in natural language
    dates = df_query['DateStr'].tolist()
    dates = [datetime.strptime(date, '%Y-%m-%d').strftime('%B %d, %Y') for date in dates]
    df_query['DateNLP'] = dates

    df_query.drop(columns=['RelName', 'QuadEventCode', 'QuadEventName', 'Docid', 'Docids'], inplace=True)

    # create query id
    df_query['QueryId'] = range(1, len(df_query) + 1)
    # put in the first column
    cols = df_query.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    df_query = df_query[cols]

    df_query.to_csv(os.path.join(output_path, 'relation_query.csv'), sep='\t', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate query for dataset')

    parser.add_argument("--dataset", type=str, default='test', help="test, test_subset")

    args = parser.parse_args()
    dataset = args.dataset

    DATA_DIR = '../data/MIRAI'

    # load query data for query list
    output_path = os.path.join(DATA_DIR, dataset)
    data_path = os.path.join(DATA_DIR, dataset, '{}_kg.csv'.format(dataset))

    df = pd.read_csv(data_path, sep='\t', dtype=str)
    df = df.drop_duplicates(subset=['QuadEventCode'])

    # load full data for full answers
    full_data_path = os.path.join(DATA_DIR, 'data_kg.csv')
    full_df = pd.read_csv(full_data_path, sep='\t', dtype=str)
    full_df = full_df[full_df['DateStr'] >= '2023-11-01']
    full_df.drop_duplicates(subset=['QuadEventCode'], inplace=True)

    # generate relation query and object query for events in full_df
    dates = full_df['DateStr'].tolist()
    actor1 = full_df['Actor1CountryCode'].tolist()
    actor2 = full_df['Actor2CountryCode'].tolist()
    relations = full_df['EventBaseCode'].tolist()
    rel_query_codes = []
    for i in range(len(dates)):
        date = dates[i]
        a1 = actor1[i]
        a2 = actor2[i]
        r = relations[i]
        rel_query_code = '({}, {}, ?, {})'.format(date, a1, a2)
        rel_query_codes.append(rel_query_code)
    full_df['RelQueryCode'] = rel_query_codes

    # generate query to answer dataframes
    generate_relation_query(df, full_df, output_path)