import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.neural_network import MLPClassifier,MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

DATA = pd.read_csv('FinalTrain.csv')
X = DATA.iloc[:,1:-1].values
Y = DATA.iloc[:,-1].values
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=0) 


mlp = MLPRegressor(hidden_layer_sizes=(16,32,16))
mlp.fit(X_train,Y_train)
Ypred = mlp.predict(X_test)
df=pd.DataFrame({'Actual':Y_test, 'Predicted':Ypred}) 
print(df[100:135])
