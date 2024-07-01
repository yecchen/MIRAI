import pandas as pd
import os
from tqdm import tqdm
from collections import defaultdict

START_DATE = 202300 # dataset start date in format of yyyymm
END_DATE = 202311 # dataset end date in format of yyyymm

DATA_DIR = '../data/kg_raw'

COL_NAMES = [ # total 61 columns
    'GlobalEventID', 'Day', 'MonthYear', 'Year', 'FractionDate', # 5 event date attributes
    'Actor1Code', 'Actor1Name', 'Actor1CountryCode', 'Actor1KnownGroupCode', 'Actor1EthnicCode', 'Actor1Religion1Code', 'Actor1Religion2Code', 'Actor1Type1Code', 'Actor1Type2Code', 'Actor1Type3Code', # 10 actor1 attributes
    'Actor2Code', 'Actor2Name', 'Actor2CountryCode', 'Actor2KnownGroupCode', 'Actor2EthnicCode', 'Actor2Religion1Code', 'Actor2Religion2Code', 'Actor2Type1Code', 'Actor2Type2Code', 'Actor2Type3Code', # 10 actor2 attributes
    'IsRootEvent', 'EventCode', 'EventBaseCode', 'EventRootCode', 'QuadClass', 'GoldsteinScale', 'NumMentions', 'NumSources', 'NumArticles', 'AvgTone', # 10 event action attributes
    'Actor1Geo_Type', 'Actor1Geo_Fullname', 'Actor1Geo_CountryCode', 'Actor1Geo_ADM1Code', 'Actor1Geo_ADM2Code', 'Actor1Geo_Lat', 'Actor1Geo_Long', 'Actor1Geo_FeatureID', # 8 actor1 geography
    'Actor2Geo_Type', 'Actor2Geo_Fullname', 'Actor2Geo_CountryCode', 'Actor2Geo_ADM1Code', 'Actor2Geo_ADM2Code', 'Actor2Geo_Lat', 'Actor2Geo_Long', 'Actor2Geo_FeatureID', # 8 actor2 geography
    'EventGeo_Type', 'EventGeo_Fullname', 'EventGeo_CountryCode', 'EventGeo_ADM1Code', 'EventGeo_ADM2Code', 'EventGeo_Lat', 'EventGeo_Long', 'EventGeo_FeatureID', # 8 event geography
    'DATEADDED', 'SOURCEURL' # 2 other event information
]


def merge_csv_files(csv_files):
    dfs = []
    for idx, csv_file in tqdm(enumerate(csv_files), total=len(csv_files)):
        try:
            df = pd.read_csv(os.path.join(DATA_DIR, csv_file), sep='\t', header=None, dtype=str)
        except:
            continue
        if len(df.columns) != 61:
            continue
        dfs.append(df)
    df = pd.concat(dfs)
    df.columns = COL_NAMES
    df.drop_duplicates(subset='GlobalEventID', inplace=True)
    return df

def load_txt_dict(lines):
    dict_a2b, dict_b2a = {}, {}
    for line in lines:
        line = line.strip()
        a, b = line.split('\t')
        dict_a2b[a] = b
        dict_b2a[b] = a
    return dict_a2b, dict_b2a



if __name__ == "__main__":

    output_directory = '../data/kg_tmp'
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # merge all 15min files to one file and save the raw data
    csv_files = os.listdir(DATA_DIR)
    df = merge_csv_files(csv_files)
    df.to_csv(os.path.join(output_directory, 'kg_raw.csv'), index=False, sep='\t')
    print(f'kg_raw.csv saved, length: {len(df)}')

    # check event date and the news date should be the same, and save the same date data
    df['NewsDate'] = [x[:8] for x in df['DATEADDED']]
    all_df = df.loc[df['Day'] == df['NewsDate']]
    all_df.to_csv(os.path.join(output_directory, 'kg_samedate.csv'), index=False, sep='\t')
    print(f'kg_samedate.csv saved, length: {len(all_df)}')

    # start cleaning kg data
    # keep the earliest added date for each URL
    dict_url2date = defaultdict(set)
    urls = all_df['SOURCEURL'].tolist()
    dates = all_df['NewsDate'].tolist()
    for idx, url in tqdm(enumerate(urls), total=len(urls)):
        dict_url2date[url].add(dates[idx])

    dict_url2date_unique = {}
    for url, dates in dict_url2date.items():
        dict_url2date_unique[url] = min(dates)

    all_df['URLday'] = [dict_url2date_unique[x] for x in all_df['SOURCEURL']]
    all_df_url = all_df.query('NewsDate == URLday')
    all_df_url.to_csv(os.path.join(output_directory, 'kg_urldate.csv'), index=False, sep='\t')
    print(f'kg_urldate.csv saved, length: {len(all_df_url)}')

    # standardize actor name and event type
    # filter out actors without country code
    dict_iso2country, dict_country2iso =load_txt_dict(open('../data/info/ISO_country_GeoNames.txt', 'r').readlines())
    all_df_info = all_df_url[all_df_url[['Actor1CountryCode', 'Actor2CountryCode']].notnull().all(1)]
    all_df_info = all_df_info[all_df_info['Actor1CountryCode'].isin(dict_iso2country.keys())]
    all_df_info = all_df_info[all_df_info['Actor2CountryCode'].isin(dict_iso2country.keys())]
    # remove self-loops (s=o)
    all_df_info = all_df_info[all_df_info['Actor1CountryCode'] != all_df_info['Actor2CountryCode']]
    # map actor country code to actual country/region
    all_df_info['Actor1CountryName'] = [dict_iso2country[x] for x in all_df_info['Actor1CountryCode']]
    all_df_info['Actor2CountryName'] = [dict_iso2country[x] for x in all_df_info['Actor2CountryCode']]

    # filter out event type unavailable events
    all_df_info = all_df_info[all_df_info['EventRootCode'] != '--']
    # standardize the CAMEO code
    dict_cameo2name, dict_name2cameo = load_txt_dict(open('../data/info/CAMEO_relation.txt', 'r').readlines())
    all_df_info['RelName'] = [dict_cameo2name[x] for x in all_df_info['EventBaseCode']]

    # standardize the date format to 'YYYY-MM-DD'
    dates = all_df_info['URLday'].tolist()
    datestrs = [x[:4] + '-' + x[4:6] + '-' + x[6:] for x in dates]
    all_df_info['DateStr'] = datestrs

    # generate event string
    actor1_codes = all_df_info['Actor1CountryCode'].tolist()
    actor2_codes = all_df_info['Actor2CountryCode'].tolist()
    actor1_names = all_df_info['Actor1CountryName'].tolist()
    actor2_names = all_df_info['Actor2CountryName'].tolist()
    rel_codes = all_df_info['EventBaseCode'].tolist()
    rel_names = all_df_info['RelName'].tolist()

    eventcodes = [', '.join([datestrs[i], actor1_codes[i], rel_codes[i], actor2_codes[i]]) for i in range(len(datestrs))]
    eventnames = [', '.join([datestrs[i], actor1_names[i], rel_names[i], actor2_names[i]]) for i in range(len(datestrs))]
    eventfullstrs = [', '.join([datestrs[i], actor1_codes[i] + ' - ' + actor1_names[i], rel_codes[i] + '-' + rel_names[i], actor2_codes[i] + '-' + actor2_names[i]]) for i in range(len(datestrs))]

    all_df_info['QuadEventCode'] = eventcodes
    all_df_info['QuadEventName'] = eventnames
    all_df_info['QuadEventFullStr'] = eventfullstrs
    all_df_info.to_csv(os.path.join(output_directory, 'kg_info.csv'), index=False, sep='\t')
    print(f'kg_info.csv saved, length: {len(all_df_info)}')



