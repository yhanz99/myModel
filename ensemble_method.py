#https://medium.com/analytics-vidhya/ensemble-modelling-in-a-simple-way-386b6cbaf913
import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn.ensemble import AdaBoostClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

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

#to store all the useful information from each model for ensemble method
estimators=[]
accuracys=[]


#adaboost
#implementationof AdaBoost
adaboost = AdaBoostClassifier(n_estimators=10, base_estimator=None, learning_rate=2, random_state=3)
estimators.append(("AdaBoost", adaboost))
adaboost.fit(X_train,Y_train)
#testing the model
Y_pred1 = adaboost.predict(X_test)
#display result of adaboost
cm = confusion_matrix(Y_test,Y_pred1)
accuracy1 = float(cm.diagonal().sum())/len(Y_test)
print("Accuracy Of AdaBoost For The Given Dataset : ", accuracy1)
#storing accuracy
adaboost_acc = accuracy_score(Y_test, Y_pred1)
accuracys.append(adaboost_acc)


#SVM
#implementation of SVM
svm_RBFkernel = SVC(kernel = 'rbf')
estimators.append(("svm_RBFkernel", svm_RBFkernel))
svm_RBFkernel.fit(X_train,Y_train)
#testing the model
Y_pred2 = svm_RBFkernel.predict(X_test)
#display result of adaboost
cm = confusion_matrix(Y_test,Y_pred2)
accuracy2 = float(cm.diagonal().sum())/len(Y_test)
print("Accuracy Of Support Vector Machine (SVM) For The Given Dataset : ", accuracy2)
#storing accuracy
svm_RBFkernel_acc = accuracy_score(Y_test, Y_pred2)
accuracys.append(svm_RBFkernel_acc)


#ensemble method
#implementation of SVM
ensemble_model = VotingClassifier(estimators)
ec = ensemble_model.fit(X_train,Y_train)
Y_pred_All = ensemble_model.predict(X_test)
ensemble_model_acc=accuracy_score(Y_test, Y_pred_All)
print("Accuracy Of Ensemble Method For The Given Dataset : ", ensemble_model_acc)


#summary
print("******************************")
print("Prediction Report: ")
print("\nConfusion Matrix\n", confusion_matrix(Y_test, Y_pred_All))
print("\n", classification_report(Y_test, Y_pred_All))


