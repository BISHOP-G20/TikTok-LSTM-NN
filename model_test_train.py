import tensorflow as tf
import pandas as pd
import numpy as np
import os
import io
import time
from tqdm import tqdm
from PIL import Image
from matplotlib import pyplot
from Word2Vec import text_to_vset
from keras.optimizers import Adam
from keras.losses import MeanSquaredError, MeanAbsoluteError
from keras.models import Model
from keras.saving import load_model
from keras.callbacks import ModelCheckpoint
from keras.layers import Input, LSTM, Dense, Concatenate

def create_model(learn_rate, lstm_units, dense1_units, dense2_units, dense3_units, num_outputs):
    # Define input layers
    input_data = Input(shape=(None,300), name='input_data')
    additional_features = Input(shape=(3), name='additional_features')

    # LSTM layer
    lstm_output = LSTM(units=lstm_units, return_sequences=False)(input_data)

    # Concatenate LSTM output with additional features
    concatenated_features = Concatenate(axis=-1)([lstm_output, additional_features])

    # Dense layers
    dense1 = Dense(units=dense1_units, activation='relu')(concatenated_features)
    dense2 = Dense(units=dense2_units, activation='relu')(dense1)
    dense3 = Dense(units=dense3_units, activation='relu')(dense2)

    # Output layer
    output = Dense(units=num_outputs, name='output')(dense3)

    # Define model inputs and outputs
    model = Model(inputs=[input_data, additional_features], outputs=output)
    model.compile(optimizer=Adam(learning_rate=learn_rate), loss=MeanSquaredError(), metrics=[MeanAbsoluteError()])

    return model

def fig2img(fig): 
    buf = io.BytesIO() 
    fig.savefig(buf) 
    buf.seek(0) 
    img = Image.open(buf) 
    return img 

def plot_model(loss, mae, model_name):
    fig, axis = pyplot.subplots(2)
    x = np.arange(0,len(loss))
    axis[0].plot(x,loss)
    axis[0].set_title(model_name + ": loss")

    axis[1].plot(x,mae)
    axis[1].set_title(model_name + ": MAE")
    pyplot.tight_layout(pad=2)
    plt = pyplot.gcf()
    img = fig2img(plt)
    img.save('images/'+model_name+'_image.png')
    #pyplot.show()

# Mid-training Preprocessing functions
def preprocess_text(text):
    return text_to_vset(text)

# Model Structure Parameters
lstm_units = 175
dense1_units = 150
dense2_units = 100
dense3_units = 50
num_outputs = 5

# Hyper Parameters
learn_rate = 0.000004
num_epochs = 40
size_train_data = 1000

# Create and Compile Model
#model = create_model(learn_rate, lstm_units,
#                    dense1_units,dense2_units,
#                    dense3_units,num_outputs)

model = load_model("C:/Users/George Bishop/VsCodeProjects/TikTok-LSTM-CNN/models/model_e11-4.keras")

# Setup Model Saving Policy
model_name = 'model_e11-5.keras'
save_path = 'C:/Users/George Bishop/VsCodeProjects/TikTok-LSTM-CNN/models/' + model_name

train_larray = [[0] * num_epochs]
train_marray = [[0] * num_epochs]
start_time = time.time()
test_loss = 0
test_mae = 0

rand_state = np.random.randint(999999999, size=num_epochs)

 # Setup CSV as Tensorflow Dataset
dataset = pd.read_csv('data/std_tiktok_data.csv', usecols=[
                                        'claim_status','video_duration_sec',
                                        'video_transcription_text','verified_status',
                                        'video_view_count','video_like_count',
                                        'video_share_count','video_download_count',
                                        'video_comment_count'], nrows=1200)
dataset = dataset.sample(frac=1, random_state=rand_state[8])
text_df = dataset['video_transcription_text']
additional_df = dataset[['claim_status','video_duration_sec','verified_status']]
label_df = dataset[['video_view_count','video_like_count','video_share_count','video_download_count','video_comment_count']]
dataset = None

# Preprocessing
vec_file_name = "train-numpy-1200-sX.npy"
vec_save_path = 'C:/Users/George Bishop/VsCodeProjects/TikTok-LSTM-CNN/data/' + vec_file_name

print('Data Preprocessing and Saving')
with open (vec_save_path, 'wb') as f:
    for i in tqdm(range(1000)):
        vector_set = preprocess_text(text_df.iloc[i])
        vector_set = vector_set.reshape(1,vector_set.shape[0],vector_set.shape[1])
        additional_set = np.array(additional_df.iloc[i]).reshape(1,3)
        label_set = np.array(label_df.iloc[i]).reshape(1,num_outputs)
        np.save(f, vector_set)
        np.save(f, additional_set)
        np.save(f, label_set)
    f.close()

    
test_vec_file_name = "test-numpy-1200-sX.npy"
test_vec_save_path = 'C:/Users/George Bishop/VsCodeProjects/TikTok-LSTM-CNN/data/' + test_vec_file_name
with open (test_vec_save_path, 'wb') as f:
    for i in tqdm(range(1000,1200)):
        vector_set = preprocess_text(text_df.iloc[i])
        vector_set = vector_set.reshape(1,vector_set.shape[0],vector_set.shape[1])
        additional_set = np.array(additional_df.iloc[i]).reshape(1,3)
        label_set = np.array(label_df.iloc[i]).reshape(1,num_outputs)
        np.save(f, vector_set)
        np.save(f, additional_set)
        np.save(f, label_set)
    f.close()

text_df = None
additional_df = None
label_df = None

model.save(save_path)

for i in range(num_epochs):
    print("\t\tSTART EPOCH " + str(i))
    epoch_start = time.time()
    with open (vec_save_path, 'rb') as f:
        for j in tqdm(range(size_train_data)):
            vector_set = np.load(f)
            additional_set = np.load(f)
            label_set = np.load(f)
            model.fit([vector_set, additional_set],label_set,epochs=1,verbose=0)
            metrics = model.get_metrics_result()
            train_larray[0][i] += float(metrics['loss'])
            train_marray[0][i] += float(metrics['mean_absolute_error'])
    print('\nEpoch: ' + str(i) + '/' + str(num_epochs)+ " | Time: " + str((time.time() - epoch_start)/60) + 'min | loss: ' + str(train_larray[0][i]) + " - MAE: " + str(train_marray[0][i]))

model.save(save_path)
end_time = time.time()

with open (test_vec_save_path, 'rb') as f:
    for i in tqdm(range(200)):
        vector_set = np.load(f)
        additional_set = np.load(f)
        label_set = np.load(f)
        model.evaluate([vector_set, additional_set],label_set, steps=1, verbose=0)
        metrics = model.get_metrics_result()
        test_loss += float(metrics['loss'])
        test_mae += float(metrics['mean_absolute_error'])


plot_model(train_larray[0], train_marray[0], model_name)

print("total time: " + str(end_time - start_time))
print(train_larray)
print(train_marray)
print("train loss: " + str(train_larray[0][num_epochs-1]/5))
print("train MAE: " + str(train_marray[0][num_epochs-1]/5))
print("test loss: " + str(test_loss))
print("test MAE: " + str(test_mae))
print(rand_state[8])

#print(model.predict([sample_txt,sample_features]))
#model.summary()
