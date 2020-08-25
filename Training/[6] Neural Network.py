import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.layers import LeakyReLU,Dropout
from keras.optimizers import Adam

DATA = pd.read_csv('FinalTrainS.csv')

X = DATA.iloc[:,1:-1].values
Y = DATA.iloc[:,-1].values

model = Sequential()
model.add(Dense(50, input_dim=5, activation='tanh'))
model.add(Dense(50))
model.add(LeakyReLU(alpha=0.2))
model.add(Dense(75))
model.add(LeakyReLU(alpha=0.1))
model.add(Dense(75))
model.add(LeakyReLU(alpha=0.2))
model.add(Dropout(0.5))
model.add(Dense(100))
model.add(LeakyReLU(alpha=0.1))
model.add(Dense(100))
model.add(LeakyReLU(alpha=0.1))
model.add(Dropout(0.5))
model.add(Dense(75))
model.add(LeakyReLU(alpha=0.1))
model.add(Dense(75))
model.add(LeakyReLU(alpha=0.1))
model.add(Dense(75))
model.add(LeakyReLU(alpha=0.1))
model.add(Dropout(0.5))
model.add(Dense(50))
model.add(LeakyReLU(alpha=0.1))
model.add(Dense(50))
model.add(LeakyReLU(alpha=0.2))
model.add(Dense(50))
model.add(LeakyReLU(alpha=0.1))
model.add(Dropout(0.5))
model.add(Dense(25))
model.add(LeakyReLU(alpha=0.1))
model.add(Dense(1,kernel_initializer='normal'))
model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.0001))
model.fit(X,Y, epochs=60000)

model.save('DocModel.model')
model.save_weights('DocWeights.h5')
