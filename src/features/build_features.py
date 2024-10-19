import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from DataTransformation import LowPassFilter, PrincipalComponentAnalysis
from TemporalAbstraction import NumericalAbstraction
from FrequencyAbstraction import FourierTransformation


# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------

df = pd.read_pickle("../../data/interim/02_data_outliers_removed.pkl")

plt.style.use('seaborn-v0_8')
plt.rcParams['figure.figsize'] = (20,5)
plt.rcParams['figure.dpi'] = 100
plt.rcParams['lines.linewidth'] = 2
# --------------------------------------------------------------
# Dealing with missing values (imputation)
# --------------------------------------------------------------

for col in df.columns:
    df[col] = df[col].infer_objects().interpolate()

# df[df['set']==51]['acc_x'].plot()



# --------------------------------------------------------------
# Calculating set duration
# --------------------------------------------------------------

# duration = df[df['set']==set].index[-1] - df[df['set']==set].index[0]
# type(duration.seconds)

net_duration = 0
for set_ in df['set'].unique():

    duration = df[df['set']==set_].index[-1] - df[df['set']==set_].index[0]
    # print(f"Duration of set {set}: {duration.seconds} seconds")
    
    net_duration += duration.seconds

    df.loc[df['set']==set_, 'duration'] = duration.seconds

# duration_df = df.groupby(['label','category'])['duration'].mean()
# print(f"Net duration of all sets: {net_duration} seconds")



# duration_df
# duration_df.iloc[0] /5
# duration_df.iloc[1] / 10
# duration_df.iloc[2] /5
# duration_df.iloc[3] / 10
# --------------------------------------------------------------
# Butterworth lowpass filter
# --------------------------------------------------------------

butterworth = LowPassFilter()
sensor_columns = list(df.columns[:6])
fs = 1000/200

butterworth_df = df.copy()
# butterworth_df = butterworth.low_pass_filter(
#         data_table = butterworth_df,
#         col=sensor_columns[0],
#         sampling_frequency=fs,
#         cutoff_frequency=1.3,
#         order=5,
#                                     )
 
# axes, fig = plt.subplots(2,1 , figsize=(20,10))

# butterworth_df[butterworth_df['set']==25]['acc_x'].plot(ax=fig[0])
# butterworth_df[butterworth_df['set']==25]['acc_x_lowpass'].plot(ax=fig[1])

for col in sensor_columns:
    butterworth_df = butterworth.low_pass_filter(
        data_table = butterworth_df,
        col=col,
        sampling_frequency=fs,
        cutoff_frequency=1.3,
        order=5,
                                    )

    butterworth_df[col] = butterworth_df[col+'_lowpass']
    butterworth_df.drop(columns=[col+'_lowpass'], inplace=True)
    



# ---
# -----------------------------------------------------------
# Principal component analysis PCA
# --------------------------------------------------------------
pca = PrincipalComponentAnalysis()
pca_df = butterworth_df.copy()
# pca_values = pca.determine_pc_explained_variance(butterworth_df, sensor_columns)

# plt.plot(pca_values)
# plt.xlabel('Principal Component')
# plt.ylabel('Explained Variance')

pca_df = pca.apply_pca(pca_df, sensor_columns, 3)

# --------------------------------------------------------------
# Sum of squares attributes
# --------------------------------------------------------------


df_squared = pca_df.copy()
df_squared['acc_r'] = df_squared[sensor_columns[:3]].apply(np.square).apply(np.sum, axis=1).apply(np.sqrt)
df_squared['gyc_r'] = df_squared[sensor_columns[3:6]].apply(np.square).apply(np.sum, axis=1).apply(np.sqrt)


# subset = df_squared[df_squared['set']==25]

# subset[['acc_r','gyc_r']].plot(subplots=True)
# --------------------------------------------------------------
# Temporal abstraction
# --------------------------------------------------------------

temporal_df = df_squared.copy()

from TemporalAbstraction import NumericalAbstraction
NBS = NumericalAbstraction()

windos_size = int(1000/200)

predictor_columns = sensor_columns + ['acc_r','gyc_r']


for set_ in df_squared['set'].unique():
    
    temp_df  , new_columns  = NBS.abstract_numerical(temporal_df[temporal_df["set"]== set_].copy(), predictor_columns, windos_size , 'mean')
    
    
    temporal_df.loc[temporal_df["set"]==set_ , new_columns] = temp_df[new_columns]

    
    temp_df2 , new_columns = NBS.abstract_numerical(temporal_df[temporal_df["set"]== set_].copy(), sensor_columns, windos_size , 'std')

    temporal_df.loc[temporal_df["set"]==set_, new_columns] = temp_df2[new_columns]


# set_ = 25
# temp_df  , new_columns  = NBS.abstract_numerical(temporal_df[temporal_df["set"]== set_].copy(), predictor_columns, windos_size , 'mean')

# temporal_df.loc[temporal_df["set"]==set_ , new_columns] = temp_df[new_columns]

    
# temporal_df.isna().sum()
# df_squared.isna().sum()

# temporal_df.columns
# --------------------------------------------------------------
# Frequency features
# --------------------------------------------------------------

freq_df = temporal_df.copy()
fs = int(1000/200)
ws  =int( 2800/200)
FTS = FourierTransformation()

df_freq_list = []

for s in freq_df['set'].unique():
    
    subset = freq_df[freq_df['set']==s].reset_index(drop=True).copy()

    subset = FTS.abstract_frequency(subset, predictor_columns, ws, fs)

    df_freq_list.append(subset)

freq_df=pd.concat(df_freq_list)











# --------------------------------------------------------------
# Dealing with overlapping windows
# --------------------------------------------------------------

# removing half of the data for better generalisation 

freq_df = freq_df.dropna()

freq_df.isna().sum()
freq_df = freq_df.iloc[::2]

# --------------------------------------------------------------
# Clustering
# --------------------------------------------------------------

from sklearn.cluster import KMeans

df_cluster = freq_df.copy()
cluster_cloumns = ['acc_x','acc_y','acc_z']
inertia = []

for k in range(2,10):
    kmeans = KMeans(n_clusters=k,n_init=20,  random_state=0)
    cluster_labels = kmeans.fit_predict(df_cluster[cluster_cloumns])
    inertia.append(kmeans.inertia_)


plt.plot(range(2,10), inertia , figure=plt.figure(figsize=(20,20)))

kmeans = KMeans(n_clusters=5,n_init=20,  random_state=0)
df_cluster['cluster'] = kmeans.fit_predict(df_cluster[cluster_cloumns])



# plot the clusters
fig, ax = plt.subplots( figsize=(20, 20))
ax = fig.add_subplot( projection='3d')

for c in df_cluster['cluster'].unique():
    subset = df_cluster[df_cluster['cluster']==c]
    ax.scatter(subset['acc_x'], subset['acc_y'], subset['acc_z'], label=f'Cluster {c}')

ax.set_xlabel('acc_x')
ax.set_ylabel('acc_y')
ax.set_zlabel('acc_z')
plt.legend()
plt.show()

# plot the clusters
fig, ax = plt.subplots( figsize=(20, 20))
ax = fig.add_subplot( projection='3d')

for l in df_cluster['label'].unique():
    subset = df_cluster[df_cluster['label']==l]
    ax.scatter(subset['acc_x'], subset['acc_y'], subset['acc_z'], label=f'Label {l}')

ax.set_xlabel('acc_x')
ax.set_ylabel('acc_y')
ax.set_zlabel('acc_z')
plt.legend()
plt.show()

# --------------------------------------------------------------
# Export dataset
# --------------------------------------------------------------

df_cluster.to_pickle("../../data/processed/03_data_features.pkl")