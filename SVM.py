#https://hands-on.cloud/implementation-of-support-vector-machine-svm-using-python/
import pandas as pd #for lading and performing preprocessing
import numpy as np
from matplotlib.colors import ListedColormap
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
pd.options.mode.chained_assignment = None

#import dataset (my_dataset)
my_dataset = pd.read_csv("diabetes_data_upload.csv")

for col in my_dataset:
    my_dataset.loc[my_dataset[col] == 'Male', col] = 0
    my_dataset.loc[my_dataset[col] == 'Female', col] = 1
    my_dataset.loc[my_dataset[col] == 'No', col] = 0
    my_dataset.loc[my_dataset[col] == 'Yes', col] = 1
    my_dataset.loc[my_dataset[col] == 'Negative', col] = 0
    my_dataset.loc[my_dataset[col] == 'Positive', col] = 1

df = pd.DataFrame(my_dataset)
df = df.astype(float)

#spliting dataset into training and testing set
training_set, test_set = train_test_split(df, test_size=0.2, random_state=3) #test size is 0.2, train size is 0.8

#appplying features from previous features selection
X_train = training_set.iloc[:, [3, 2, 1, 4, 12]].values
#target "class" in dataset
Y_train = training_set.iloc[:,16].values
#appplying features from previous features selection
X_test = test_set.iloc[:,[3, 2, 1, 4, 12]].values
#target "class" in dataset
Y_test = test_set.iloc[:,16].values

#implementation of SVM
svm_RBFkernel = SVC(kernel = 'rbf')
svm_RBFkernel.fit(X_train,Y_train)

#testing the model
y_pred = svm_RBFkernel.predict(X_test)

#displaying accuracy
print (accuracy_score(Y_test, y_pred))

