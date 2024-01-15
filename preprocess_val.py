import pandas as pd
import os
import numpy as np

bio_entity = "rna"
main_dir = '/Users/mathieugierski/Nextcloud/Macbook M3/Oncopole/data/test'
saving_path = '/Users/mathieugierski/Nextcloud/Macbook M3/Oncopole/test_'+bio_entity+'.csv'

means_path = '/Users/mathieugierski/Nextcloud/Macbook M3/Oncopole/mean_'+bio_entity+'.csv'
mini_path = '/Users/mathieugierski/Nextcloud/Macbook M3/Oncopole/min_'+bio_entity+'.csv'
maxi_path = '/Users/mathieugierski/Nextcloud/Macbook M3/Oncopole/max_'+bio_entity+'.csv'
path_remove = '/Users/mathieugierski/Nextcloud/Macbook M3/Oncopole/removed_'+bio_entity+'.txt'


def load_and_concatenate_csv(folder_path):
    all_data = pd.DataFrame()
    i=0
    f=-1

    for root, dirs, files in os.walk(folder_path):

        i=0
        for file in files:
            if file.endswith('.csv') and bio_entity in root:

                if i==0:
                    f+=1

                file_path = os.path.join(root, file)
                df = pd.read_csv(file_path, index_col=0)

                df.columns.values[0] = str(f) + "_"+str(i)

                #print(df.T.iloc[0])

                all_data = pd.concat([all_data, df.T])
                #print(str(f) + "_"+str(i))
                print(all_data.shape)
                i+=1
            
        
    print(all_data.shape)
    #print(all_data)
    return all_data

def replace_nones_with_mean(dataframe):
    print("replacing")

    with open(path_remove, 'r') as file:
        columns_to_remove = [line.strip() for line in file]

    #print(columns_to_remove)

    dataframe.drop(columns_to_remove, axis=1, inplace=True)

    means = pd.read_csv(means_path)
    return dataframe.fillna(means).fillna(0)

def min_max_normalize(dataframe):
    print("min-max")

    mini = pd.read_csv(mini_path)
    maxi = pd.read_csv(maxi_path)

    if bio_entity=="dnam":
        mini.set_index('Composite Element REF', inplace=True)
        maxi.set_index('Composite Element REF', inplace=True)

    elif bio_entity=="rna":
        mini.set_index('gene_id', inplace=True)
        maxi.set_index('gene_id', inplace=True)

    return (dataframe - mini.iloc[:, 0].values)/(maxi.iloc[:, 0].values-mini.iloc[:, 0].values)

# Folder path
folder_path = '/Users/mathieugierski/Nextcloud/Macbook M3/Oncopole/data/test'


data = load_and_concatenate_csv(folder_path)
data = replace_nones_with_mean(data)
print(data.isna().any().any())
print(data.shape)
data = min_max_normalize(data)
print(data.shape)
#print(normalized_data)

data.to_csv(saving_path)