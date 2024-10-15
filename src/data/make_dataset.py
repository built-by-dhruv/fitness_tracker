import pandas as pd
from glob import glob
import os 





def read_metamotion_data(file_path):
    acc_df = pd.DataFrame()
    gyr_df = pd.DataFrame()

    acc_set = 1
    gyr_set = 1


    all_files = glob(file_path)    
    for file in all_files:

        individial_file_meta_data = file.split("/")[4].split("\\")[1].split("-")
        participant  =  individial_file_meta_data[0]
        label     = individial_file_meta_data[1]
        category       = individial_file_meta_data[2].rstrip("123").rstrip("_MetaWear_2019")

        data = pd.read_csv(file)

        data["participant"] = participant
        data["label"] = label
        data["category"] = category

        if "Accelerometer" in individial_file_meta_data[-1]:
            data["set"] = acc_set
            acc_set += 1
            acc_df = pd.concat([acc_df, data], axis=0)
        if "Gyroscope" in individial_file_meta_data[-1]:
            data["set"] = gyr_set
            gyr_set += 1
            gyr_df = pd.concat([gyr_df, data], axis=0)	

    acc_df.index = pd.to_datetime(acc_df["epoch (ms)"] , unit="ms")
    gyr_df.index = pd.to_datetime(gyr_df["epoch (ms)"] , unit="ms")

    acc_df.drop(["epoch (ms)" , "time (01:00)","elapsed (s)"], axis=1, inplace=True)
    gyr_df.drop(["epoch (ms)" , "time (01:00)","elapsed (s)"], axis=1, inplace=True)

    return acc_df, gyr_df 

all_files = r"../../data/raw/MetaMotion/*.csv"
acc_df, gyr_df = read_metamotion_data(all_files)


# acc_df.iloc[:,:3]
# data_merged = pd.merge(acc_df, gyr_df, on=["participant", "label", "category", "set"], how="outer")
data_merged = pd.concat([acc_df.iloc[:,:3], gyr_df], axis=1)
# data_merged.head(1000)
# data_merged.to_csv("../../data/processed/data_merged.csv")
# list(data_merged.columns)
data_merged.columns = [
    'acc_x',
    'acc_y',
    'acc_z',
    'gyc_x',
    'gyc_y',
    'gyc_z',
    'participant',
    'label',
    'category',
    'set']

# --------------------------------------------------------------
# Resample data (frequency conversion)
# --------------------------------------------------------------
sampling = {
    'acc_x':'mean',
    'acc_y':'mean',
    'acc_z':'mean',
    'gyc_x':'mean',
    'gyc_y':'mean',
    'gyc_z':'mean',
    'participant':'last',
    'label':'last',
    'category':'last',
    'set':'last'
}
# data_merged = data_merged.resample(rule="200ms").agg(sampling)

days = [ g for n, g in data_merged.groupby(pd.Grouper(freq="D")) ]
# day[0]

data_resampled = pd.concat([data.resample(rule="200ms").agg(sampling).dropna() for data in days])
data_resampled.sample(1000) 
# data_resampled.to_csv("../../data/processed/01_data_processed.csv")
data_resampled.info()

# put data in pickel file
data_resampled.to_pickle("../../data/interim/01_data_processed.pkl"  )

# --------------------------------------------------------------
# Export dataset
# --------------------------------------------------------------
