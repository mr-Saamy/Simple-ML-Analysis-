# %%
"""
# Data preprocessing
"""

# %%
import pandas as pd

# %%
"""
# Read Dataset
"""

# %%
data = pd.read_csv("Iris.csv")

# %%
"""
# Analysis of data
"""

# %%
data.head(10)

# %%
data.tail(10)

# %%
"""
# Cleaning Data 
"""

# %%
data.drop('Id',axis=1) #drop(column name, inplace='True'(permanent deletion) , axis=1(column)....) Temporary deletion

# %%
data.head()

# %%
data.drop('Id', inplace=True, axis=1)

# %%
data.head() #permanent deletion

# %%
"""
# Check missing Data
"""

# %%
data['Species'].value_counts()#to find data available

# %%
"""
# Visualize a count plot for Species 
"""

# %%
import seaborn as sb
import matplotlib as mpl

# %%
sb.countplot(data['Species']) #Visualization of no of species

# %%
data['Species'].unique() #show unique categories

# %%
"""
# Selection of Algorithm
"""

# %%
# Setosa-0 | Versi-1 | Virgi-2
classes = {'Iris-setosa':0, 'Iris-versicolor':1, 'Iris-virginica':2} #dictionary assigninh keyvalues to species
data.replace({'Species':classes}, inplace=True) #replace species with values from classes

# %%
data['Species'].unique() #show unique categories

# %%
sb.pairplot(data,hue='Species')

# %%
"""
# Create Arrays
"""

# %%
x=data.iloc[:,0:4].values
y=data.iloc[:,-1].values

# %%
"""
# Training Dataset
"""

# %%
from sklearn.model_selection import train_test_split as tts

# %%
x_train ,x_test ,y_train ,y_test = tts(x,y,test_size=0.2, random_state=12)

# %%
print(y_train.shape)

# %%
print(y_test.shape)

# %%
"""
# Logistic Regression
"""

# %%
from sklearn.linear_model import LogisticRegression as lr

# %%
model_lr=lr()

# %%
model_lr.fit(x_train,y_train)

# %%
"""


# Check Accuracy
"""

# %%
model_lr.score(x_test,y_test)

# %%
"""
# Standard Scaler
"""

# %%
#mean = 0
#SD = 1
#Normalizing
from sklearn.preprocessing import StandardScaler as ss

# %%
scaler = ss()
scaler.fit(x_train)

# %%
x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)

# %%
"""
# KNN Algorithm
"""

# %%
from sklearn.neighbors import KNeighborsClassifier as knn
mknn=knn()

# %%
mknn.fit(x_train, y_train)

# %%
mknn.score(x_test,y_test)

# %%
