# Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")
#To check Performances
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
def Performance(actual_value , predicted_value):
    accuracy = accuracy_score(actual_value , predicted_value) * 100
    precision = precision_score(actual_value , predicted_value) * 100
    recall = recall_score(actual_value , predicted_value) * 100
    f1 = f1_score(actual_value , predicted_value, average='weighted')
    print('Accuracy is {:.4f}%\n Precision is {:.4f}%\n Recall is {:.4f}%\nF1 Score is {:.4f}\n'.format(accuracy, precision, recall, f1))
filepath = r'C:\major project\kaggle_data\\'
file= filepath+'training_data_2_csv_UTF.csv'

training_data = pd.read_csv(file)
bots = training_data[training_data.bot==1]
nonbots = training_data[training_data.bot==0]
training_data.columns
training_data.head(10)
training_data.describe()
training_data = training_data.replace(r'^\"|\"$', '', regex=True)

# Specify numeric columns to be converted
numeric_columns = [
    'id', 'followers_count', 'friends_count', 'listed_count', 'statuses_count',
    'favourites_count', 'verified', 'default_profile', 'default_profile_image', 'has_extended_profile', 'bot'
]

# Convert specified columns to numeric, coercing errors to NaN
for col in numeric_columns:
    training_data[col] = pd.to_numeric(training_data[col], errors='coerce')

# Drop rows with any NaN values in the specified numeric columns
training_data = training_data.dropna(subset=numeric_columns)

# Select only the numeric columns for correlation
features = [
    'id', 'followers_count', 'friends_count', 'listed_count', 'statuses_count',
    'favourites_count', 'verified', 'default_profile', 'default_profile_image', 'has_extended_profile', 'bot'
]

# Calculate the correlation matrix using Spearman method
correlation_matrix = training_data[features].corr(method='spearman')

# Generate a mask for the upper triangle
mask = np.zeros_like(correlation_matrix, dtype=np.bool_)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(16, 12))

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(correlation_matrix, mask=mask, cmap='coolwarm', vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True)

plt.title('Correlation Matrix of Features')
plt.show()
training_data.drop(['id_str', 'screen_name', 'location', 'description', 'url', 'created_at', 'lang', 'status', 'has_extended_profile','name'],axis=1,inplace=True)
training_data.head()
sns.catplot(x="bot", y="followers_count", data=training_data);
sns.catplot(x="bot", y="friends_count", data=training_data);
sns.catplot(x="bot", y="listed_count", data=training_data);
sns.catplot(x="bot", y="favourites_count", data=training_data);
sns.catplot(x="bot", y="verified", data=training_data);
sns.catplot(x="bot", y="statuses_count", data=training_data);
X = training_data.iloc[:, :-1].values
y = training_data.iloc[:, 9].values
from sklearn.preprocessing import LabelEncoder
Labelx=LabelEncoder()
X[:,5]=Labelx.fit_transform(X[:,5])
X[:,7]=Labelx.fit_transform(X[:,7])
X[:,8]=Labelx.fit_transform(X[:,8])
from sklearn.neighbors import KNeighborsClassifier as knn
classifier=knn(n_neighbors=5)
classifier.fit(X,y)
bots = training_data[training_data.bot==1]
Nbots = training_data[training_data.bot==0]
 
B = bots.iloc[:,:-1]
B_y = bots.iloc[:,9]
B_pred = classifier.predict(B)

#Confusionmatrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(B_y,B_pred)
Performance(B_y,B_pred)

NB = Nbots.iloc[:,:-1]
NB_y = Nbots.iloc[:,9]
NB_pred = classifier.predict(NB)

#Confusionmatrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(NB_y,NB_pred)
Performance(NB_y,NB_pred)
from sklearn.svm import SVC
classifier=SVC(kernel='rbf', random_state=0)
classifier.fit(X,y)
bots = training_data[training_data.bot==1]
Nbots = training_data[training_data.bot==0]
 
B = bots.iloc[:,:-1]
B_y = bots.iloc[:,9]
B_pred = classifier.predict(B)

#Confusionmatrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(B_y,B_pred)
Performance(B_y,B_pred)

NB = Nbots.iloc[:,:-1]
NB_y = Nbots.iloc[:,9]
NB_pred = classifier.predict(NB)

#Confusionmatrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(NB_y,NB_pred)
Performance(NB_y,NB_pred)

from sklearn.naive_bayes import GaussianNB as GNB
classifier=GNB()
classifier.fit(X,y)
bots = training_data[training_data.bot==1]
Nbots = training_data[training_data.bot==0]
 
B = bots.iloc[:,:-1]
B_y = bots.iloc[:,9]
B_pred = classifier.predict(B)

#Confusionmatrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(B_y,B_pred)
Performance(B_y,B_pred)

NB = Nbots.iloc[:,:-1]
NB_y = Nbots.iloc[:,9]
NB_pred = classifier.predict(NB)

#Confusionmatrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(NB_y,NB_pred)
Performance(NB_y,NB_pred)

#fitting
from sklearn.ensemble import RandomForestClassifier as rf
classifier= rf(n_estimators=10,criterion='entropy',random_state=0)
classifier.fit(X,y)
bots = training_data[training_data.bot==1]
Nbots = training_data[training_data.bot==0]

B = bots.iloc[:,:-1]
B_y = bots.iloc[:,9]
B_pred = classifier.predict(B)

#Confusionmatrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(B_y,B_pred)
Performance(B_y,B_pred)

NB = Nbots.iloc[:,:-1]
NB_y = Nbots.iloc[:,9]
NB_pred = classifier.predict(NB)
 
#Confusionmatrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(NB_y,NB_pred)
Performance(NB_y,NB_pred)

#fitting
from sklearn.ensemble import RandomForestClassifier as rf
classifier= rf(n_estimators=10,criterion='entropy',random_state=0)
classifier.fit(X,y)
bots = training_data[training_data.bot==1]
Nbots = training_data[training_data.bot==0]

B = bots.iloc[:,:-1]
B_y = bots.iloc[:,9]
B_pred = classifier.predict(B)

#Confusionmatrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(B_y,B_pred)
Performance(B_y,B_pred)

NB = Nbots.iloc[:,:-1]
NB_y = Nbots.iloc[:,9]
NB_pred = classifier.predict(NB)
 
#Confusionmatrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(NB_y,NB_pred)
Performance(NB_y,NB_pred)

from sklearn.tree import DecisionTreeClassifier as DTC
classifier= DTC(criterion="entropy")
classifier.fit(X,y)
bots = training_data[training_data.bot==1]
Nbots = training_data[training_data.bot==0]
 
B = bots.iloc[:,:-1]
B_y = bots.iloc[:,9]
B_pred = classifier.predict(B)

#Confusionmatrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(B_y,B_pred)
Performance(B_y,B_pred)

NB = Nbots.iloc[:,:-1]
NB_y = Nbots.iloc[:,9]
NB_pred = classifier.predict(NB)

#Confusionmatrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(NB_y,NB_pred)
Performance(NB_y,NB_pred)

from sklearn.tree import DecisionTreeClassifier as DTC
classifier= DTC(criterion="entropy")
classifier.fit(X,y)
filepath = r'C:\major project\kaggle_data\\'
test_data_path = filepath + 'test_data_4_students.csv'
test_data = pd.read_csv(test_data_path, sep='\t', encoding='ISO-8859-1');
test_data.drop(['id_str', 'screen_name', 'location', 'description', 'url', 'created_at', 'lang', 'status', 'has_extended_profile','name'],axis=1,inplace=True)


X1 = training_data.iloc[:, :-1].values

from sklearn.preprocessing import LabelEncoder
Labelx=LabelEncoder()
X1[:,6]=Labelx.fit_transform(X1[:,6])
X1[:,7]=Labelx.fit_transform(X1[:,7])
X1[:,8]=Labelx.fit_transform(X1[:,8])
y1_pred = classifier.predict(X1)
y1_pred=pd.DataFrame(y1_pred);
y1_pred.to_csv("Result.csv",index=False);

print(y1_pred)

