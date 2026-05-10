#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import pandas as pd


# # Pobieranie danych

# In[2]:


tf.keras.utils.get_file(
    "bike_sharing_dataset.zip",
    "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip",
    cache_dir=".",
    extract=True
)


# # Przygotowanie danych

# In[3]:


df = pd.read_csv('datasets/bike_sharing_dataset_extracted/hour.csv')
df['datetime'] = pd.to_datetime(df['dteday'] + ' ' + df['hr'].astype(str).str.zfill(2), format='%Y-%m-%d %H')
df.set_index('datetime', inplace=True)


# In[4]:


print((df.index.min(), df.index.max()))


# In[5]:


(365 + 366) * 24 - len(df)


# In[6]:


df


# In[7]:


count_cols = ['casual', 'registered', 'cnt']
weather_cols = ['temp', 'atemp', 'hum', 'windspeed']
categorical_cols = ['holiday', 'weekday', 'workingday', 'weathersit']


# In[8]:


frequency = 'h'

counts_regular = df[count_cols].resample(frequency).asfreq().fillna(0)
weather_regular = df[weather_cols].resample(frequency).interpolate(method='time')
categorical_regular = df[categorical_cols].resample(frequency).ffill()


# In[9]:


df = pd.concat([counts_regular, weather_regular, categorical_regular], axis=1)


# In[10]:


df


# In[11]:


df.notna().sum()


# In[12]:


df[['casual', 'registered', 'cnt', 'weathersit']].describe()


# In[13]:


df.casual /= 1e3
df.registered /= 1e3
df.cnt /= 1e3
df.weathersit /= 4


# In[14]:


df_2weeks = df[:24 * 7 * 2]
df_2weeks[['casual', 'registered', 'cnt', 'temp']].plot(figsize=(10, 3))


# In[15]:


df_daily = df.resample('W').mean()
df_daily[['casual', 'registered', 'cnt', 'temp']].plot(figsize=(10, 3))


# # Wskaźniki bazowe

# In[16]:


df_1 = df.copy()

df_1['pred_daily'] = df_1['cnt'].shift(24)
mae_daily = (df_1['cnt'] - df_1['pred_daily']).abs().mean() * 1000

df_1['pred_weekly'] = df_1['cnt'].shift(24 * 7)
mae_weekly = (df_1['cnt'] - df_1['pred_weekly']).abs().mean() * 1000


# In[17]:


(mae_daily, mae_weekly)


# In[18]:


import pickle


# In[19]:


with open('mae_baseline.pkl', 'wb') as f:
    pickle.dump((mae_daily, mae_weekly), f)


# # Predykcja przy pomocy sieci gęstej

# In[20]:


cnt_train = df['cnt']['2011-01-01 00:00':'2012-06-30 23:00']
cnt_valid = df['cnt']['2012-07-01 00:00':]


# In[21]:


seq_len = 1 * 24
train_ds = tf.keras.utils.timeseries_dataset_from_array(
    cnt_train.to_numpy(),
    targets=cnt_train[seq_len:],
    sequence_length=seq_len,
    batch_size=32,
    shuffle=True,
    seed=42
)
valid_ds = tf.keras.utils.timeseries_dataset_from_array(
    cnt_valid.to_numpy(),
    targets=cnt_valid[seq_len:],
    sequence_length=seq_len,
    batch_size=32
)


# In[22]:


model = tf.keras.Sequential([
    tf.keras.Input(shape=(seq_len,)),
    tf.keras.layers.Dense(1)
])


# In[23]:


model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.Huber(),
    metrics=['mae']
)


# In[24]:


history_dense = model.fit(train_ds, epochs=20, validation_data=valid_ds, verbose=1)


# In[25]:


model.save('model_linear.keras')


# In[26]:


mae_linear = model.evaluate(valid_ds)[1] * 1000


# In[27]:


mae_linear


# In[28]:


with open('mae_linear.pkl', 'wb') as f:
    pickle.dump((mae_linear,), f)


# # Prosta sieć rekurencyjna

# In[29]:


model_rnn1 = tf.keras.Sequential([
    tf.keras.Input(shape=[None, 1]),
    tf.keras.layers.LSTM(1)
])


# In[30]:


model_rnn1.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.Huber(),
    metrics=['mae']
)


# In[31]:


history_rnn1 = model_rnn1.fit(train_ds, epochs=20, validation_data=valid_ds, verbose=1)


# In[32]:


model_rnn1.save('model_rnn1.keras')


# In[33]:


mae_rnn1 = model_rnn1.evaluate(valid_ds)[1] * 1000


# In[34]:


mae_rnn1


# In[35]:


with open('mae_rnn1.pkl', 'wb') as f:
    pickle.dump((mae_rnn1,), f)


# In[36]:


model_rnn32 = tf.keras.Sequential([
    tf.keras.Input(shape=[None, 1]),
    tf.keras.layers.LSTM(32),
    tf.keras.layers.Dense(1)
])


# In[37]:


model_rnn32.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.Huber(),
    metrics=['mae']
)


# In[38]:


history_rnn32 = model_rnn32.fit(train_ds, epochs=20, validation_data=valid_ds, verbose=1)


# In[39]:


model_rnn32.save('model_rnn32.keras')


# In[40]:


mae_rnn32 = model_rnn32.evaluate(valid_ds)[1] * 1000


# In[41]:


mae_rnn32


# In[42]:


with open('mae_rnn32.pkl', 'wb') as f:
    pickle.dump((mae_rnn32,), f)


# # Głęboka RNN

# In[43]:


model_rnn_deep = tf.keras.Sequential([
    tf.keras.Input(shape=[None, 1]),
    tf.keras.layers.LSTM(512, return_sequences=True),
    tf.keras.layers.LSTM(512, return_sequences=True),
    tf.keras.layers.LSTM(512),
    tf.keras.layers.Dense(1)
])


# In[44]:


model_rnn_deep.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.Huber(),
    metrics=['mae']
)


# In[45]:


history_rnn_deep = model_rnn_deep.fit(train_ds, epochs=20, validation_data=valid_ds, verbose=1)


# In[46]:


model_rnn_deep.save('model_rnn_deep.keras')


# In[47]:


mae_rnn_deep = model_rnn_deep.evaluate(valid_ds)[1] * 1000


# In[48]:


mae_rnn_deep


# In[49]:


with open('mae_rnn_deep.pkl', 'wb') as f:
    pickle.dump((mae_rnn_deep,), f)


# # Model wielowymiarowy

# In[50]:


feature_cols = ['cnt', 'weathersit', 'atemp', 'workingday']
target_col = 'cnt'

features_df = df[feature_cols]
target_series = df[target_col]

train_features = features_df.loc[:'2012-06-30']
valid_features = features_df.loc['2012-07-01':]
train_targets = target_series.loc[:'2012-06-30']
valid_targets = target_series.loc['2012-07-01':]


# In[51]:


train_ds_mv = tf.keras.utils.timeseries_dataset_from_array(
    train_features.to_numpy(),
    targets=train_targets[seq_len:],
    sequence_length=seq_len,
    batch_size=32,
    shuffle=True,
    seed=42
)
valid_ds_mv = tf.keras.utils.timeseries_dataset_from_array(
    valid_features.to_numpy(),
    targets=valid_targets[seq_len:],
    sequence_length=seq_len,
    batch_size=32
)


# In[52]:


model_rnn_mv = tf.keras.Sequential([
    tf.keras.Input(shape=[None, 4]),
    tf.keras.layers.LSTM(32),
    tf.keras.layers.Dense(1)
])


# In[53]:


model_rnn_mv.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.Huber(),
    metrics=['mae']
)


# In[54]:


history_rnn_mv = model_rnn_mv.fit(train_ds_mv, epochs=20, validation_data=valid_ds_mv, verbose=1)


# In[55]:


model_rnn_mv.save('model_rnn_mv.keras')


# In[56]:


mae_rnn_mv = model_rnn_mv.evaluate(valid_ds_mv)[1] * 1000


# In[57]:


mae_rnn_mv


# In[58]:


with open('mae_rnn_mv.pkl', 'wb') as f:
    pickle.dump((mae_rnn_mv,), f)

