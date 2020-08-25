import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
DATA = pd.read_csv('Emtrain.csv')

Quals = DATA.iloc[:,1].values
Exps= DATA.iloc[:,2].values
Ratings = DATA.iloc[:,3].values
Places = DATA.iloc[:,4].values
Profiles = DATA.iloc[:,5].values
Fees = DATA.iloc[:,-1].values
sc = StandardScaler()
# Encoding Qualifications:
MAP = {}
MAP['MD'] = 180
MAP['MBBS'] = 170
MAP['BDS'] = 70
MAP['G.A.M.S'] = 50
MAP['BHMS'] = 80
MAP['Get inspired by remarkable stories of people like you'] = 20
MAP['DLO'] = 150
MAP['PhD'] = 140
MAP['DDVL'] = 60
MAP['DDV'] = 40
MAP['MS'] = 190
MAP['DNB'] = 110
MAP['Diploma in Dermatology'] = 30
MAP['LCEH'] = 100
MAP['GCEH'] = 90
MAP['BSAM'] = 160
MAP['BAMS'] = 120
Qualsn = []

for i in range(len(Quals)):
	temp = Quals[i]
	ans = MAP[temp]
	Qualsn.append(ans)




print(Qualsn[:10])
Qualsns = np.array(Qualsn).reshape(-1,1)
scaler = sc.fit(Qualsns)
QualsS = scaler.fit_transform(Qualsns)
# Encoding of Qualifications Finished!


# Encoding Places
MAP = {}
MAP['Ernakulam'] = 80
MAP['Chennai'] = 60
MAP['Coimbatore'] = 70
MAP['Unknown'] = 30
MAP['Delhi'] = 90
MAP['Thiruvananthapuram'] = 50
MAP['Mumbai'] = 120
MAP['Bangalore'] = 110
MAP['Hyderabad'] = 100

Placesn = []

for i in range(len(Places)):
	temp = Places[i]
	ans = MAP[temp]
	Placesn.append(ans)



print(Placesn[:10])
Placesns = np.array(Placesn).reshape(-1,1)
scaler = sc.fit(Placesns)
PlacesS = scaler.fit_transform(Placesns)
# Encoding of Places Done!


#Now, Encoding the Profiles

MAP = {}
MAP['Homeopath'] = 70
MAP['Ayurveda'] = 130
MAP['General Medicine'] = 150
MAP['Dentist'] = 100
MAP['Dermatologists'] = 110
MAP['ENT Specialist'] = 170

Profilesn= []

for i in range(len(Profiles)):
	temp = Profiles[i]
	ans = MAP[temp]
	Profilesn.append(ans)


print(Profilesn[:10])
Profilesns = np.array(Profilesn).reshape(-1,1)
scaler = sc.fit(Profilesns)
ProfilesS = scaler.fit_transform(Profilesns)
# Profiles encoded Too!


Expsns = np.array(Exps).reshape(-1,1)
scaler = sc.fit(Expsns)
ExpsS = scaler.fit_transform(Expsns)



Ratingsns = np.array(Ratings).reshape(-1,1)
scaler = sc.fit(Ratingsns)
RatingS = scaler.fit_transform(Ratingsns)

Flist = []

for index in range(len(Quals)):
	temp = []
	temp.append(QualsS[index][0])
	temp.append(ExpsS[index][0])
	temp.append(RatingS[index][0])
	temp.append(PlacesS[index][0])
	temp.append(ProfilesS[index][0])
	temp.append(Fees[index])

	Flist.append(temp)

Cols = ['Qualification','Experiece','Ratings','Place','Profile','Fees']
df = pd.DataFrame(Flist, columns = Cols)

df.to_csv('FinalTrainS.csv') 










