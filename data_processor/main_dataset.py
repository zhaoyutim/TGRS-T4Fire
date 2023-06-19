import pandas as pd
from data_processor.DatesetProcessor import DatasetProcessor
dfs = []
for year in ['2018', '2019', '2020']:
    filename = '/geoinfo_vol1/home/z/h/zhao2/CalFireMonitoring/roi/us_fire_' + year + '_out_new.csv'
    df = pd.read_csv(filename)
    dfs.append(df)
df = pd.concat(dfs, ignore_index=True)

if __name__ == '__main__':
    satellites = ['VIIRS_Day']
    val_ids = ['24462610', '24462788', '24462753']
    test_ids = ['24461623', '24332628']
    skip_ids = ['21890069', '20777160', '20777163', '20777166']

    df = df.sort_values(by=['Id'])
    df['Id'] = df['Id'].astype(str)

    train_df = df[~df.Id.isin(val_ids + skip_ids + test_ids)]
    val_df = df[df.Id.isin(val_ids)]
    test_df = df[df.Id.isin(test_ids)]

    train_ids = train_df['Id'].values.astype(str)
    val_ids = val_df['Id'].values.astype(str)
    test_ids = test_df['Id'].values.astype(str)

    proj2_processor = DatasetProcessor()
    proj2_processor.dataset_generator_proj2_image(train_ids, file_name ='proj3_all_fire_img_v3.npy')