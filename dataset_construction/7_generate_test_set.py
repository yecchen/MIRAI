import os
import numpy as np
import pandas as pd

DATA_DIR = '../data/MIRAI'
output_dir = '../data/MIRAI/test'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


if __name__ == "__main__":
    df = pd.read_csv(os.path.join(DATA_DIR, 'data_kg.csv'), sep='\t', dtype=str)
    data_news = pd.read_csv(os.path.join(DATA_DIR, 'data_news.csv'), sep='\t', dtype=str)
    data_final = pd.read_csv(os.path.join(DATA_DIR, 'data_final.csv'), sep='\t', dtype=str)

    print(len(df))
    df_nov = df[df['DateStr'] >= '2023-11-01']
    print(len(df_nov))
    df_nov_uniq = df_nov.drop_duplicates(subset=['QuadEventCode'])
    len(df_nov_uniq)

    # add regional information
    # load continent data
    df_continent = pd.read_csv('../data/info/continents2.csv', sep=',')
    # create region and sub-region column to the data_nov_uniq
    dict_iso2region = dict(zip(df_continent['alpha-3'], df_continent['region']))
    dict_iso2subregion = dict(zip(df_continent['alpha-3'], df_continent['sub-region']))
    df_nov_uniq['Actor1Region'] = df_nov_uniq['Actor1CountryCode'].map(dict_iso2region)
    df_nov_uniq['Actor1SubRegion'] = df_nov_uniq['Actor1CountryCode'].map(dict_iso2subregion)
    df_nov_uniq['Actor2Region'] = df_nov_uniq['Actor2CountryCode'].map(dict_iso2region)
    df_nov_uniq['Actor2SubRegion'] = df_nov_uniq['Actor2CountryCode'].map(dict_iso2subregion)

    # add event root code column
    df_nov_uniq['EventRootCode'] = df_nov_uniq['EventBaseCode'].apply(lambda x: x[:2])

    # add query column
    dates = df_nov_uniq['DateStr'].values
    actor1 = df_nov_uniq['Actor1CountryCode'].values
    actor2 = df_nov_uniq['Actor2CountryCode'].values
    query = []
    for date, a1, a2 in zip(dates, actor1, actor2):
        query.append(f"{date} {a1} {a2}")
    df_nov_uniq['query'] = query

    # filter test set by a higher threshold of daily mentions (100)
    dict_quadcode2mention = dict(zip(data_final['QuadEventCode'], data_final['NumDailyMentions']))
    df_nov_uniq['NumDailyMentions'] = df_nov_uniq['QuadEventCode'].map(dict_quadcode2mention)
    df_nov_uniq['NumDailyMentions'] = df_nov_uniq['NumDailyMentions'].astype(int)
    df_nov_uniq = df_nov_uniq[df_nov_uniq['NumDailyMentions'] >= 100]

    # filter test set by a threshold of downloadable news articles (5)
    # get number of news distribution per event
    docids = df_nov_uniq['Docids'].values
    docids = [eval(docid) for docid in docids]
    quedeventnames = df_nov_uniq['QuadEventName'].values

    dict_event2news = {}
    for event, docid in zip(quedeventnames, docids):
        if event not in dict_event2news:
            dict_event2news[event] = []
        dict_event2news[event] += docid

    # remove duplicates
    for event in dict_event2news:
        dict_event2news[event] = list(set(dict_event2news[event]))

    # get number of news per query
    dict_event2news_count = {event: len(docid) for event, docid in dict_event2news.items()}
    # get a distribution of news count, use describe to get the distribution
    df_nov_uniq['news_count'] = df_nov_uniq['QuadEventName'].map(dict_event2news_count)
    dict_query2mincount = {}
    for query, count in zip(df_nov_uniq['query'], df_nov_uniq['news_count']):
        if query not in dict_query2mincount:
            dict_query2mincount[query] = count
        else:
            dict_query2mincount[query] = min(dict_query2mincount[query], count)
    df_nov_uniq['min_news_count'] = df_nov_uniq['query'].map(dict_query2mincount)
    df_above_mean = df_nov_uniq[(df_nov_uniq['min_news_count'] >= 5)]

    # add answer columns
    dict_query2ans = {}
    for query, code in zip(df_nov_uniq['query'], df_nov_uniq['EventBaseCode']):
        if query not in dict_query2ans:
            dict_query2ans[query] = [code]
        else:
            dict_query2ans[query].append(code)
    for query in dict_query2ans:
        dict_query2ans[query] = list(set(dict_query2ans[query]))
    df_nov_uniq['answer'] = df_nov_uniq['query'].map(dict_query2ans)
    df_nov_uniq['answer_len'] = df_nov_uniq['answer'].apply(lambda x: len(x))

    # save test set
    df_above_mean.to_csv(os.path.join(output_dir, 'test_kg.csv'), sep='\t', index=False)
    print(f'test_kg.csv saved, length: {len(df_above_mean)}')

