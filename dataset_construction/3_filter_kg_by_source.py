import os
import json
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

START_DATE = 202300 # dataset start date in format of yyyymm
END_DATE = 202311 # dataset end date in format of yyyymm

DATA_DIR = '../data/kg_tmp'
output_directory = '../data/kg_tmp'
text_output_directory = '../data/text_tmp'
if not os.path.exists(text_output_directory):
    os.makedirs(text_output_directory)


def plot_cumu_plot(df):
    # define aesthetics for plot
    color1 = 'steelblue'
    color2 = 'red'
    line_size = 4

    # create basic bar plot
    fig, ax = plt.subplots()
    ax.bar(df.index, df['QuadCount'], color=color1)

    # add cumulative percentage line to plot
    ax2 = ax.twinx()
    ax2.plot(df.index, df['QuadCumPercent'], color=color2, marker="D", ms=line_size)
    ax2.yaxis.set_major_formatter(PercentFormatter())

    # specify axis colors
    ax.tick_params(axis='y', colors=color1)
    ax2.tick_params(axis='y', colors=color2)

    # display Pareto chart
    plt.show()


if __name__ == "__main__":
    df = pd.read_csv(os.path.join(DATA_DIR, 'kg_info.csv'), sep='\t', dtype=str)
    df['NumMentions'] = df['NumMentions'].astype(int)

    # ensure date range
    df = df.sort_values(by='Day', kind='mergesort')
    df = df[(df['MonthYear'] >= str(START_DATE)) & (df['Day'] <= str(END_DATE))]

    # check daily mention of each event, plot the distribution and decide the minimum threshold
    dict_quad2daymention = defaultdict(int)
    quads = df['QuadEventCode'].to_list()
    mentions = df['NumMentions'].to_list()
    for _, idx in tqdm(enumerate(range(len(quads))), total=len(quads)):
        dict_quad2daymention[quads[idx]] += mentions[idx]
    df['NumDailyMentions'] = [dict_quad2daymention[quad] for quad in quads]

    dict_mention_count = defaultdict(int)
    for quad, daymention in dict_quad2daymention.items():
        dict_mention_count[daymention] += 1

    total_counts = sum(dict_mention_count.values())
    mention_df = pd.DataFrame.from_dict({'NumDailyMentions': dict_mention_count.keys(), 'QuadCount': dict_mention_count.values()})
    mention_df.sort_values(by='NumDailyMentions', ascending=False, inplace=True, ignore_index=True)
    mention_df['QuadCumCount'] = mention_df['QuadCount'].cumsum()
    mention_df['QuadCumPercent'] = mention_df['QuadCumCount'] / total_counts
    plot_cumu_plot(mention_df)

    df_mentionfilter = df[df['NumDailyMentions'] > 50]

    sorted_df = df_mentionfilter.sort_values(by='Day', kind='mergesort')
    sorted_df.to_csv(os.path.join(output_directory, 'kg_source.csv'), index=False, sep='\t')
    print(f'kg_source.csv saved, length: {len(sorted_df)}')

    # save news urls
    uniq_urls = df['SOURCEURL'].unique().tolist()
    json.dump(open(os.path.join(text_output_directory, 'unique_urls.json'), 'w'), uniq_urls)
    print(f'unique_urls.json saved, length: {len(uniq_urls)}')




