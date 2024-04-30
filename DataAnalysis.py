import pandas as pd
import numpy as np
df = pd.read_csv('MetroPT3_AirCompressor.csv')
print(df.head())

list_of_column_names = df.columns.tolist()
print(list_of_column_names)

df_to_numpy = df.to_numpy()
period = df_to_numpy[:,0]
timestamp = df_to_numpy[:,1]
# -------------- Anomaly periods ------------------
anomaly_ts_1_start = '2020-04-18 00:00:01' #tested
anomaly_ts_1_stop = '2020-04-18 23:59:56'  #tested

anomaly_ts_2_start = '2020-05-29 23:30:08' #tested
anomaly_ts_2_stop = '2020-05-30 06:00:04'  #tested

anomaly_ts_3_start = '2020-06-05 10:00:04'#tested
anomaly_ts_3_stop = '2020-06-07 14:19:39' #tested: there is no sensor data from 14:19:40 until 2020-06-08 11:48:04

anomaly_ts_4_start = '2020-07-15 14:30:00' #tested
anomaly_ts_4_stop = '2020-07-15 19:00:00'  #tested
#----------------------------------------------------


# -------------Extracting anomaly sets-----------------

# Anomaly Set 1
start_index = np.where(timestamp == anomaly_ts_1_start)[0]
start_index1 = int(start_index)
stop_index = np.where(timestamp == anomaly_ts_1_stop)[0]
stop_index1 = int(stop_index)
anomaly_set_1 = df_to_numpy[start_index1:stop_index1+1,:]

# Anomaly Set 2
start_index = np.where(timestamp == anomaly_ts_2_start)[0]
start_index2 = int(start_index)
stop_index = np.where(timestamp == anomaly_ts_2_stop)[0]
stop_index2 = int(stop_index)
anomaly_set_2 = df_to_numpy[start_index2:stop_index2+1,:]

# Anomaly Set 3
start_index = np.where(timestamp == anomaly_ts_3_start)[0]
start_index3 = int(start_index)
stop_index = np.where(timestamp == anomaly_ts_3_stop)[0]
stop_index3 = int(stop_index)
anomaly_set_3 = df_to_numpy[start_index3:stop_index3+1,:]

# Anomaly Set 4
start_index = np.where(timestamp == anomaly_ts_4_start)[0]
start_index4 = int(start_index)
stop_index = np.where(timestamp == anomaly_ts_4_stop)[0]
stop_index4 = int(stop_index)
anomaly_set_4 = df_to_numpy[start_index4:stop_index4+1,:]

anamaly_num = len(anomaly_set_1)+len(anomaly_set_2)+len(anomaly_set_3)+len(anomaly_set_4)
normal_num = len(timestamp) - anamaly_num
anomaly_set = np.concatenate((anomaly_set_1,anomaly_set_2,anomaly_set_3,anomaly_set_4),axis = 0)
print(anomaly_set.shape)

print("\n normal_num ",normal_num)
print("\n anamaly_num: ",anamaly_num)
print("\n Total: ",normal_num+anamaly_num)
#-------------------------------------------------------------

#  -------------------Normal Set---------------------------------------------------
normal_set = df_to_numpy[0:start_index1]
normal_set = np.append(normal_set,df_to_numpy[stop_index1+1:start_index2,:],axis = 0)
normal_set = np.append(normal_set,df_to_numpy[stop_index2+1:start_index3,:],axis = 0)
normal_set = np.append(normal_set,df_to_numpy[stop_index3+1:start_index4,:],axis = 0)
normal_set = np.append(normal_set,df_to_numpy[stop_index4+1:-1], axis = 0)
# ----------------------------------------------------------------------------------
normal_set_continuous = normal_set[:,[2,3,4,5,6,7,8]]
anomaly_set_continuous = anomaly_set[:,[2,3,4,5,6,7,8]]
#-----------------------------------------------------------------------------------

# ---------Histogram for features in anomaly dataset -------------------------------
import matplotlib.pyplot as plt

# Sample NumPy array (replace with your actual data)
data = anomaly_set_continuous

# Number of features (columns)
num_features = data.shape[1]

# Create a subplot grid for plotting
fig, axes = plt.subplots(nrows=1, ncols=num_features, figsize=(12, 4))
features = []

# Plot histogram for each feature (column)
for i in range(num_features):
  axes[i].hist(data[:, i],bins = 40)  # data[:, i] selects the i-th column
  axes[i].set_title(f"Feature {i+1}")  # Customize titles if needed
  axes[i].set_xlabel(list_of_column_names[i+2])
  axes[i].set_ylabel("Frequency")

# Adjust layout and display the plot
fig.suptitle("Histogram of Each Feature in the Anomaly Data")
plt.tight_layout()
plt.show()
# -----------------------------------------------------------------------------

# ------- Histogram of features in normal dataset -----------------------------
# Sample NumPy array (replace with your actual data)
data = normal_set_continuous

# Number of features (columns)
num_features = data.shape[1]

# Create a subplot grid for plotting
fig, axes = plt.subplots(nrows=1, ncols=num_features, figsize=(12, 4))

# Plot histogram for each feature (column)
for i in range(num_features):
  axes[i].hist(data[:, i],bins = 40)  # data[:, i] selects the i-th column
  axes[i].set_title(f"Feature {i+1}")  # Customize titles if needed
  axes[i].set_xlabel(list_of_column_names[i+2])
  axes[i].set_ylabel("Frequency")

# Adjust layout and display the plot
fig.suptitle("Histogram of Each Feature in the normal Data")
plt.tight_layout()
plt.show()
# ------------------------------------------------------------

# Normalizing normal data
normal_set_continuous = normal_set_continuous.astype(np.float64) # Converting elements to np.float64 in order to enable
                                                                 # vectorized operations in next line
# Normalize each feature (column)
normalized_data = (normal_set_continuous - np.mean(normal_set_continuous, axis=0)) / np.std(normal_set_continuous, axis=0)
# ----------------------------------------------------------------------------

# Normalizing nomaly data
anomaly_set_continuous = anomaly_set_continuous.astype(np.float64) # Converting elements to np.float64 in order to enable
                                                                  # vectorized operations in next line
# Normalize each feature (column)
anomaly_normalized_data = (anomaly_set_continuous - np.mean(anomaly_set_continuous, axis=0)) / np.std(anomaly_set_continuous, axis=0)