import os
import json
import numpy as np
import pandas as pd
from datetime import datetime


DATA_DIR = '../data/MIRAI'
output_dir = '../data/MIRAI/test_subset'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


def sample_balanced_data(df, n_samples=100, max_attempts=50, curr_subset=None):
    # Define categories to balance
    unique_dates = df['DateStr'].unique()
    unique_event_codes = df['EventRootCode'].unique()
    unique_regions = df['Actor1Region']._append(df['Actor2Region']).unique()
    unique_country = df['Actor1CountryCode']._append(df['Actor2CountryCode']).unique()

    # Initialize sampled DataFrame and counts
    sampled_df = pd.DataFrame()
    date_counts = {date: 0 for date in unique_dates}
    code_counts = {code: 0 for code in unique_event_codes}
    region_counts = {region: 0 for region in unique_regions}
    country_counts = {c: 0 for c in unique_country}

    # Pre-add a specific event (e.g., 5th row of df)
    if curr_subset is not None:
        sampled_df = pd.concat([sampled_df, curr_subset])
        curr_dates = curr_subset['DateStr'].values
        curr_codes = curr_subset['EventRootCode']._append(curr_subset['Actor2Region']).values
        curr_regions = curr_subset['Actor1Region']._append(curr_subset['Actor2Region']).values
        curr_country = curr_subset['Actor1CountryCode']._append(curr_subset['Actor2CountryCode']).values
        for i in range(len(curr_dates)):
            date_counts[curr_dates[i]] += 1
            code_counts[curr_codes[i]] += 1
            region_counts[curr_regions[i]] += 1
            country_counts[curr_country[i]] += 1

    # Helper function to calculate balance score
    def calculate_balance_score():
        date_variance = np.var(list(date_counts.values()))
        code_variance = np.var(list(code_counts.values()))
        region_variance = np.var(list(region_counts.values())) * 0.4
        country_variance = np.var(list(country_counts.values())) * 0.2
        return date_variance + code_variance + region_variance + country_variance

    # Sampling procedure with attempts to find the most balanced addition
    def balanced_sampling_step(df, sampled_df):
        best_score = float('inf')
        best_sample = None
        for _ in range(max_attempts):
            temp_sample = df.sample(n=1)
            temp_date = temp_sample['DateStr'].values[0]
            temp_code = temp_sample['EventRootCode'].values[0]
            temp_region = temp_sample['Actor1Region']._append(temp_sample['Actor2Region']).values[0]
            temp_country = temp_sample['Actor1CountryCode']._append(temp_sample['Actor2CountryCode']).values[0]

            # Simulate adding this sample to the counts
            date_counts[temp_date] += 1
            code_counts[temp_code] += 1
            region_counts[temp_region] += 1
            country_counts[temp_country] += 1

            # Calculate the balance score with this potential addition
            new_score = calculate_balance_score()

            # Undo the simulated addition
            date_counts[temp_date] -= 1
            code_counts[temp_code] -= 1
            region_counts[temp_region] -= 1
            country_counts[temp_country] -= 1

            # Check if this sample is better
            if new_score < best_score:
                best_score = new_score
                best_sample = temp_sample

        return best_sample

    # Main sampling loop
    while (len(sampled_df) == 0 or len(sampled_df['query'].unique()) < n_samples) and not df.empty:
        best_sample = balanced_sampling_step(df, sampled_df)
        best_date = best_sample['DateStr'].values[0]
        best_code = best_sample['EventRootCode'].values[0]
        best_region = best_sample['Actor1Region']._append(best_sample['Actor2Region']).values[0]
        best_country = best_sample['Actor1CountryCode']._append(best_sample['Actor2CountryCode']).values[0]

        # Update the actual data and counts
        sampled_df = pd.concat([sampled_df, best_sample])
        date_counts[best_date] += 1
        code_counts[best_code] += 1
        region_counts[best_region] += 1
        country_counts[best_country] += 1

        # Remove selected sample from the original DataFrame
        df = df.drop(best_sample.index).reset_index(drop=True)

        # Break the loop if no more rows are available
        if len(df) == 0:
            break

    return sampled_df.reset_index(drop=True)


if __name__ == "__main__":
    df_nov_uniq = pd.read_csv(os.path.join(DATA_DIR, 'test/test_kg.csv'), sep='\t', dtype=str)

    # sample a balanced subset of the data
    curr_subset = pd.DataFrame()
    sampled_data = sample_balanced_data(df_nov_uniq, n_samples=100, max_attempts=50, curr_subset=curr_subset)

    # sort by date
    sampled_data = sampled_data.sort_values(by='DateStr')
    # drop index column
    sampled_data = sampled_data.drop(columns=['index'])
    # save test subset
    sampled_data.to_csv(os.path.join(output_dir, 'test_subset_kg.csv'), sep='\t', index=False)




