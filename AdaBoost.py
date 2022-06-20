import pandas as pd #for lading and performing preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import confusion_matrix
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
training_set, test_set = train_test_split(df, test_size=0.2, random_state=1) #test size is 0.2, train size is 0.8

#appplying features from previous features selection
X_train = training_set.loc[:, ["Polydipsia", "Polyuria", "Gender", "sudden weight loss", "partial paresis"]].values
#target "class" in dataset
Y_train = training_set.iloc[:,16].values
#appplying features from previous features selection
X_test = test_set.loc[:,["Polydipsia", "Polyuria", "Gender", "sudden weight loss", "partial paresis"]].values
#target "class" in dataset
Y_test = test_set.iloc[:,16].values

#implementationof AdaBoost
adaboost = AdaBoostClassifier(n_estimators=50, base_estimator=None, learning_rate=0.5, random_state=1)
adaboost.fit(X_train,Y_train)

#testing the model
Y_pred = adaboost.predict(X_test)

cm = confusion_matrix(Y_test,Y_pred)
accuracy = float(cm.diagonal().sum())/len(Y_test)
print("Accuracy Of AdaBoost For The Given Dataset : ", accuracy)

