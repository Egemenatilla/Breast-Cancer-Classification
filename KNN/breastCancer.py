# Libraries
import pandas as pd
import numpy as np  
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV # en iyi KNN parametresi için import ettin.
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.neighbors import KNeighborsClassifier, NeighborhoodComponentsAnalysis,LocalOutlierFactor
from sklearn.decomposition import PCA


data = pd.read_csv("data.csv") # Read data from csv file
cancer = data.copy()


cancer.drop(['Unnamed: 32','id'],inplace = True, axis = 1) # We dropped unused columns

#print(cancer.head())


# Label Encoding(0-1) for diagnosis column
le = LabelEncoder()
cancer['diagnosis'] = le.fit_transform(cancer['diagnosis'])
#print(cancer['diagnosis']) # M - 1, B - 0
# ----------------

print("Veri Seti Boyutu: ",cancer.shape)
print(cancer.info())
print(cancer.describe())
print(cancer.isnull().values.any()) # No Missing Values
#%% EDA
# Correlation
corr_matrix = cancer.corr()
sns.clustermap(corr_matrix,annot = True, fmt = ".2f")
plt.title("Özellikler arasındaki korelasyon ilişkisi")
plt.show()

#%%
threshold = 0.75
filtre = np.abs(corr_matrix["diagnosis"]) > threshold
corr_features = corr_matrix.columns[filtre].tolist()
sns.clustermap(data[corr_features].corr(), annot = True,fmt = '.2f')
plt.title("Koreleasyon Eşiği 0.75 üzerinde olan featureslar arasındaki ilişki ")
#%% box Plot
data_melted = pd.melt(cancer,id_vars='diagnosis',var_name='features',value_name='value')
plt.figure()
sns.boxplot(x = 'features',y = 'value',hue = 'diagnosis', data = data_melted)
plt.xticks(rotation = 45)
plt.show()
#%% Pair Plot
sns.pairplot(cancer[corr_features],diag_kind = 'kde',markers='+',hue = 'diagnosis')
plt.show()


#%%
# Outlier Detection 

y = cancer.diagnosis
x = cancer.drop(["diagnosis"],axis = 1)
columns = x.columns.tolist()

clf = LocalOutlierFactor() # default neighbor = 20
y_pred = clf.fit_predict(x)

# print(y_pred) outliers -1
#Number of outliers
sums = 0
for i in range(0,569):
    if(y_pred[i] == -1 ):
        sums += 1
outlier = sums

print(outlier)

X_score = clf.negative_outlier_factor_

outlier_score = pd.DataFrame()
outlier_score["score"] = X_score
# threshold
threshold = -2.5    # Identifies outliers over 2.5.
filtre = outlier_score["score"] < threshold
outlier_index = outlier_score[filtre].index.tolist()

#----------
plt.figure()
plt.scatter(x.iloc[outlier_index,0],x.iloc[outlier_index,1], color = 'blue', s = 50 , label = "Data Points")
plt.scatter(x.iloc[:,0],x.iloc[:,1], color = 'k', s = 3 , label = "Data Points")

radius = (X_score.max() - X_score)/(X_score.max() - X_score.min())
outlier_score["radius"] = radius
plt.scatter(x.iloc[:,0],x.iloc[:,1], s = 1000*radius, edgecolors = "r", facecolor = "none",label = "Outlier Scores")
plt.legend()
plt.show()
#----------

# drop outliers
x = x.drop(outlier_index)
y = y.drop(outlier_index).values
#%%
# Train test split
X_train,X_test,Y_train,Y_test = train_test_split(x,y, test_size = 0.3, random_state = 42)
#%% standardization
#standardization - 

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ---box plot
X_train_df = pd.DataFrame(X_train, columns = columns)

X_train_df_describe = X_train_df.describe()


X_train_df["diagnosis"] = Y_train


data_melted = pd.melt(X_train_df,id_vars='diagnosis',var_name='features',value_name='value') #Her feature ı tek tek görmemizi sağlar

plt.figure()
sns.boxplot(x = 'features',y = 'value',hue = 'diagnosis', data = data_melted)
plt.xticks(rotation = 90)
plt.show()

# pair plot
sns.pairplot(X_train_df[corr_features],diag_kind = 'kde',markers='+',hue = 'diagnosis')
plt.show()

#%% 
# Basic KNN Algorthm
knn = KNeighborsClassifier(n_neighbors = 2)
knn.fit(X_train, Y_train)
y_pred = knn.predict(X_test)
#Confussion Matrix
cm = confusion_matrix(Y_test,y_pred)
#Accuracy
acc = accuracy_score(Y_test,y_pred)
print("Score: %",acc*100)
print("CM: ",cm)
print("Basic KNN Acc: ",acc)






