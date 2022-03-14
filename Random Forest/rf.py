# Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

# Read data from csv 
data = pd.read_csv('data.csv')

# Dropping the unused column
data.drop(["id","Unnamed: 32"],axis=1,inplace=True)

# Distribution of benign and malignant tumors
M = data[data.diagnosis == "M"]
B = data[data.diagnosis == "B"]

plt.scatter(M.radius_mean,M.texture_mean,color="red",label="bad",alpha= 0.3)
plt.scatter(B.radius_mean,B.texture_mean,color="green",label="good",alpha= 0.3)
plt.xlabel("radius_mean")
plt.ylabel("texture_mean")
plt.legend()
plt.show()

# Label Encoding
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data['diagnosis'] = le.fit_transform(data['diagnosis'])

# Standardization & Slicing
x_data = data.drop(["diagnosis"],axis=1)
x = (x_data - np.min(x_data))/(np.max(x_data)-np.min(x_data))
y = data.diagnosis.values

# Splitting of data as Training and Test
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.3,random_state=1)

# Random Forest
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators = 100,random_state = 1)
rfc.fit(x_train,y_train)

# Accuracy
prediction = rfc.predict(x_test)
score = rfc.score(x_test,y_test)
print("Accuracy of rf  %: ",score * 100)

