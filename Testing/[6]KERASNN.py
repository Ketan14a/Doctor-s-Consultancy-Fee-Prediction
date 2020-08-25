import numpy as np
import pandas as pd
from keras.models import load_model
import math


DATA = pd.read_csv('FinalTestS.csv')

X = DATA.iloc[:,1:].values
Y = []
model = load_model('14587.model')

for value in X:
	v = []
	for i in value:
		v.append(i)
	ans = model.predict(np.array([v]))
	Y.append(ans)



Results = []
for i in Y:
	temp=i[0][0]
	temp=round(temp)
	Results.append(int(temp))



Df = pd.DataFrame(Results)
Df.to_csv('Ans.csv',index=False)




