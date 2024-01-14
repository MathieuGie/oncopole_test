import pandas as pd
import os

bio_entity = ""

def load_and_concatenate_csv(folder_path):
    all_data = pd.DataFrame()
    i=0
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.csv'):
                file_path = os.path.join(root, file)
                #df_cleaned = df.loc[df.index == 'fpkm_uq_unstranded']
                df = pd.read_csv(file_path, index_col=0)

                #print(df.T.iloc[0])

                all_data = pd.concat([all_data, df.T])
                print(all_data.shape)
                print(i)
                i+=1

    print(all_data.shape)
    print(all_data)
    return all_data

def replace_nones_with_mean(dataframe):
    print("replacing")

    print(dataframe.mean(axis=0))
    return dataframe.fillna(dataframe.mean(axis=0))

def min_max_normalize(dataframe):
    print("min-max")
    return (dataframe - dataframe.min()) / (dataframe.max() - dataframe.min())

# Folder path
folder_path = '/Users/mathieugierski/Nextcloud/Macbook M3/Oncopole/data/train/'+bio_entity

# Load and concatenate data
data = load_and_concatenate_csv(folder_path)

# Replace None values with column-wise mean
data = replace_nones_with_mean(data)

# Normalize data
normalized_data = min_max_normalize(data)

#print(normalized_data)

saving_path = '/Users/mathieugierski/Nextcloud/Macbook M3/Oncopole/normalised_'+bio_entity
normalized_data.to_csv(saving_path)

print(normalized_data)
