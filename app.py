import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, SimpleRNN
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.callbacks import EarlyStopping
vac_size=1000
#loading imbd data
(X_train,Y_train),(X_test,Y_test)=imdb.load_data(num_words=vac_size)
sent_len=500
#padding making each sentance of eual length
X_train = sequence.pad_sequences(X_train, maxlen=sent_len)
X_test=sequence.pad_sequences(X_test,maxlen=sent_len)
dim=128
#creating an simple rnn model
model=Sequential()
model.add(Embedding(vac_size,dim,input_length=sent_len))
model.add(SimpleRNN(128,activation='relu'))
model.add(Dense(1,activation='sigmoid'))
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
earlystopping=EarlyStopping(monitor='val_loss',patience=5,restore_best_weights=True)
#train the model with early stopping
history=model.fit(X_train,Y_train,epochs=10,callbacks=earlystopping,validation_split=0.2)
model.save('model.h5')
