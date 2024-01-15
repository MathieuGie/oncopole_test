import pandas as pd
import os
import numpy as np

bio_entity = "rna"
saving_path = '/Users/mathieugierski/Nextcloud/Macbook M3/Oncopole/train_'+bio_entity+'.csv'
saving_path_mean = '/Users/mathieugierski/Nextcloud/Macbook M3/Oncopole/mean_'+bio_entity+'.csv'
saving_path_min = '/Users/mathieugierski/Nextcloud/Macbook M3/Oncopole/min_'+bio_entity+'.csv'
saving_path_max = '/Users/mathieugierski/Nextcloud/Macbook M3/Oncopole/max_'+bio_entity+'.csv'
saving_path_remove = '/Users/mathieugierski/Nextcloud/Macbook M3/Oncopole/removed_'+bio_entity+'.txt'

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
                df.columns.values[0] = str(i)

                all_data = pd.concat([all_data, df.T])
                print(all_data.shape)
                print(i)
                i+=1

    print(all_data.shape)
    print(all_data)
    print(all_data["ENSG00000002586.20_PAR_Y"])
    return all_data

def replace_nones_with_mean(dataframe):
    print("replacing nan")

    #Remove empty columns
    columns_to_remove = dataframe.columns[dataframe.isna().all()]
    dataframe.drop(columns_to_remove, axis=1, inplace=True)

    columns_to_remove_list = columns_to_remove.tolist()

    print(columns_to_remove)

    with open(saving_path_remove, 'w') as file:
        for column in columns_to_remove_list:
            file.write(f"{column}\n")

    #Replace none in non-fully empty columns with mean of column
    dataframe.replace(to_replace=[None], value=np.nan, inplace=True)
    return dataframe.fillna(dataframe.mean(axis=0)).fillna(0), dataframe.mean(axis=0)

def min_max_normalize(dataframe):
    print("min-max")
    normalized_df = pd.DataFrame(index=dataframe.index, columns=dataframe.columns)
    mini = dataframe.min()
    maxi = dataframe.max()
    
    # Iterate over columns
    for column in dataframe.columns:

        min_val = mini[column]
        max_val = maxi[column]
        range_val = max_val - min_val

        if range_val != 0:
            normalized_df[column] = (dataframe[column] - min_val) / range_val
        else:
            normalized_df[column] = 0.0
    
    return normalized_df, mini, maxi

# Folder path
folder_path = '/Users/mathieugierski/Nextcloud/Macbook M3/Oncopole/data/train/'+bio_entity


data = load_and_concatenate_csv(folder_path)
data, means = replace_nones_with_mean(data)
print(data.shape)
normalized_data, mini, maxi = min_max_normalize(data)
print(normalized_data.shape)

normalized_data.to_csv(saving_path)
means.to_csv(saving_path_mean)
mini.to_csv(saving_path_min)
maxi.to_csv(saving_path_max)