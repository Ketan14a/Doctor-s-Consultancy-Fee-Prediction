import pandas as pd
from sklearn.preprocessing import Imputer
import numpy as np




DATA1 = pd.read_csv('Mtest.csv')

Quals = DATA1.iloc[:,1].values

for v in range(len(Quals)):
	if "MD" in Quals[v] or "M.D." in Quals[v]:
		Quals[v]="MD"


for v in range(len(Quals)):
	if "MBBS" in Quals[v]:
		Quals[v]="MBBS"


for v in range(len(Quals)):
	if "MDS" in Quals[v]:
		Quals[v]="MDS"

for v in range(len(Quals)):
	if "DNB" in Quals[v]:
		Quals[v]="DNB"

for v in range(len(Quals)):
	if "BAMS" in Quals[v]:
		Quals[v]="BAMS"

for v in range(len(Quals)):
	if "BDS" in Quals[v]:
		Quals[v]="BDS"

for v in range(len(Quals)):
	if "BHMS" in Quals[v]:
		Quals[v]="BHMS"

for v in range(len(Quals)):
	if "DDVL" in Quals[v]:
		Quals[v]="DDVL"

DATA1.iloc[:,1] = Quals
#Done with the Qualifications

ExpData = DATA1.iloc[:,2].values
imputer = Imputer(missing_values = 0, strategy = "mean", axis = 0)
ExpData = np.array(ExpData).reshape(-1,1)
imputer = imputer.fit(ExpData)
ExpData= imputer.transform(ExpData)
DATA1.iloc[:,2] = ExpData
#Done with the Experience

RatData = DATA1.iloc[:,3].values
imputer = Imputer(missing_values = 0, strategy = "mean", axis = 0)
RatData = np.array(RatData).reshape(-1,1)
imputer = imputer.fit(RatData)
RatData= imputer.transform(RatData)
DATA1.iloc[:,3] = RatData
#Done with the Ratings

PlData = DATA1.iloc[:,4].values
# Done with the Places

ProfData = DATA1.iloc[:,5].values
#Done with the Profiles



Flist = []

for index in range(len(Quals)):
	temp = []
	temp.append(Quals[index])
	temp.append(ExpData[index][0])
	temp.append(RatData[index][0])
	temp.append(PlData[index])
	temp.append(ProfData[index])

	Flist.append(temp)


Cols = ['Qualification','Experiece','Ratings','Place','Profile']
df = pd.DataFrame(Flist, columns = Cols)

df.to_csv('Emtest.csv') 












