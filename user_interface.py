import streamlit as st  # for interface
import time
import pandas as pd  # for lading and performing preprocessing
import seaborn as sns
# https://scikit-learn.org/stable/modules/ensemble.html#:~:text=The%20sklearn.ensemble%20module%20includes%20two%20averaging%20algorithms%20based,are%20perturb-and-combine%20techniques%20%5BB1998%5D%20specifically%20designed%20for%20trees.
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.metrics import accuracy_score, confusion_matrix, plot_confusion_matrix, plot_roc_curve, precision_score, \
    recall_score
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from annotated_text import annotated_text
# https://ruslanmv.com/blog/Web-Application-Classification
# https://www.stackvidhya.com/plot-confusion-matrix-in-python-and-why/
from PIL import Image

# full view in webpage
img = Image.open('logo.png')
st.set_page_config(
    page_title="Diabetes Prediction Test",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon=img
)
st.set_option('deprecation.showPyplotGlobalUse', False)
########################################################################################################################
# model building
def build_model(DF):
    # convert words into number
    for col in DF:
        DF.loc[DF[col] == 'Male', col] = 0
        DF.loc[DF[col] == 'Female', col] = 1
        DF.loc[DF[col] == 'No', col] = 0
        DF.loc[DF[col] == 'Yes', col] = 1
        DF.loc[DF[col] == 'Negative', col] = 0
        DF.loc[DF[col] == 'Positive', col] = 1
    DF = pd.DataFrame(DF)
    # convert object into float (datatype)
    DF = DF.astype(float)
    ####################################################################################################################
    # features selection
    X_fs = DF.iloc[:, 0:16]  # independent columns
    y_fs = DF.iloc[:, -1]  # target column "class"
    # apply SelectKBest class to extract top 5 best features
    bestfeatures = SelectKBest(score_func=chi2, k=5)
    fit = bestfeatures.fit(X_fs, y_fs)
    dfscores = pd.DataFrame(fit.scores_)
    dfcolumns = pd.DataFrame(X_fs.columns)
    # concat two dataframes for better visualization
    featureScores = pd.concat([dfcolumns, dfscores], axis=1)
    featureScores.columns = ['Features', 'Score']  # naming the dataframe columns
    st.subheader("**Section 2: Features Selection**")
    st.markdown("**Top 5 features. The score indicates the importance level of each feature to the dataset.**")
    st.write(featureScores.nlargest(5, 'Score'))  # print 5 best features
    st.markdown("""<hr style="height:5px; border:none; color:#594B44; background-color:#594B44;" /> """,
                unsafe_allow_html=True)
    ####################################################################################################################
    X_df = DF.loc[:, ["Polydipsia", "Polyuria", "Gender", "Sudden Weight Loss",
                      "Partial Paresis"]]  # X=all columns except last column
    y_df = DF.iloc[:, -1]  # Y=last column

    # data splitting
    ## for adaboost
    X_train1, X_test1, y_train1, y_test1 = train_test_split(X_df, y_df, test_size=0.3,
                                                            random_state=9)  # test size is 0.3, train size is 0.7

    ## for SVM
    X_train3, X_test3, y_train3, y_test3 = train_test_split(X_df, y_df, test_size=0.25,
                                                            random_state=8)  # test size is 0.25, train size is 0.75

    ## for ensemble method
    X_train2, X_test2, y_train2, y_test2 = train_test_split(X_df, y_df, test_size=0.4,
                                                            random_state=8)  # test size is 0.4, train size is 0.6
    ####################################################################################################################
    # model performance
    st.subheader('Section 3: Model Performance')
    # #Adaboost
    # setting decision tree as decision stump
    decisiontree_clf = DecisionTreeClassifier(max_depth=1, criterion='gini', random_state=1)
    decisiontree_clf.fit(X_train1, y_train1)
    # implement decision tree into adaboost
    adaboostclf = AdaBoostClassifier(base_estimator=decisiontree_clf, n_estimators=50, learning_rate=1,
                                     algorithm='SAMME.R', random_state=1)
    #train and test the model
    adaboostclf.fit(X_train1, y_train1)
    target_pred1 = adaboostclf.predict(X_test1)
    adaboostclf_train_sc = accuracy_score(y_train1, adaboostclf.predict(X_train1))
    adaboostclf_test_sc = accuracy_score(y_test1, target_pred1) #this

    # SVM
    SVM = SVC(kernel="rbf", random_state=0)
    SVM.fit(X_train3, y_train3)
    # accuracy of SVM
    target_pred2 = SVM.predict(X_test3)
    SVM_train_sc = accuracy_score(y_train3, SVM.predict(X_train3))
    SVM_test_sc = accuracy_score(y_test3, target_pred2)

    ##ENSEMBLE METHOD
    # to store all the useful information from each model for ensemble method
    estimators = []
    accuracys = []

    # 1. Adabbost
    decisiontree_clf = DecisionTreeClassifier(max_depth=1, criterion='gini', random_state=1)
    decisiontree_clf.fit(X_train1, y_train1)
    adaboostclf = AdaBoostClassifier(base_estimator=decisiontree_clf, n_estimators=6, learning_rate=1,
                                     algorithm='SAMME.R', random_state=1)
    estimators.append(("Adaboost", adaboostclf))
    adaboostclf.fit(X_train1, y_train1)  # trains the model based on the training data.
    target_pred1 = adaboostclf.predict(X_test1)
    adaboostclf_train_sc = accuracy_score(y_train1, adaboostclf.predict(X_train1))
    adaboostclf_test_sc = accuracy_score(y_test1, target_pred1)  # this
    accuracys.append(adaboostclf_test_sc)

    # 2. SVM
    SVM = SVC(kernel="rbf", random_state=0)
    estimators.append(("svm_RBFkernel", SVM))
    SVM.fit(X_train3, y_train3)
    target_pred2 = SVM.predict(X_test3)
    SVM_train_sc = accuracy_score(y_train3, SVM.predict(X_train3))
    SVM_test_sc = accuracy_score(y_test3, target_pred2)  # this
    accuracys.append(SVM_test_sc)

    # 1+2. ensemble method
    ensemble_model = VotingClassifier(estimators)
    ec = ensemble_model.fit(X_train2, y_train2)
    target_pred3 = ec.predict(X_test2)
    ec_train_sc = accuracy_score(y_train2, ec.predict(X_train2))
    ec_test_sc = accuracy_score(y_test2, target_pred3)

    ##########################################################################################################

    # display accuracy result (Adaboost)
    annotated_text("", ("⌨Model 1: AdaBoost with Decision Tree As Base Algorithm", "", "#88cbff"))
    st.write("\n")
    st.write(" **A. The Accuracy Result of Train Set** ", (adaboostclf_train_sc.round(4)*100),"%")
    st.write(" **B. The Accuracy Result of Test Set** ", (adaboostclf_test_sc.round(4)*100),"%")
    if adaboostclf_test_sc > SVM_test_sc and adaboostclf_test_sc > ec_test_sc:
        st.success("**Conclusion** :Model 1 is the _**strongest**_ model!")
    elif adaboostclf_test_sc < SVM_test_sc and adaboostclf_test_sc < ec_test_sc:
        st.error("**Conclusion** :Model 1 is the _**weakest**_ model!")
    elif adaboostclf_test_sc > SVM_test_sc and adaboostclf_test_sc < ec_test_sc:
        st.warning("**Conclusion** :Model 1 is the _**stronger**_ than Model 2, but _**weaker**_ than Model 3!")
    elif adaboostclf_test_sc < SVM_test_sc and adaboostclf_test_sc > ec_test_sc:
        st.warning("**Conclusion** :Model 1 is the _**stronger**_ than Model 3, but _**weaker**_ than Model 2!")
    else:
        st.error("**Conclusion** :Model 1 is the _**weakest**_ model!")
    expand_Ada1 = st.expander(label='Model Evaluation: Confusion Matrix')
    with expand_Ada1:
        model1Report1_col1, model1Report1_col2 = st.columns(2)
        with model1Report1_col1:
            # confusion matrix graph
            model1_cfMatrix = confusion_matrix(y_test1, target_pred1)
            fig1CF, ax = plt.subplots(figsize=(4, 4))
            ax = sns.heatmap(model1_cfMatrix, annot=True, cmap='YlGnBu')
            ax.set_title("Diabetes Prediction")
            ax.set_ylabel("Actual Value")
            ax.set_xlabel("Predicted Value")
            ## Ticket labels - List must be in alphabetical order
            ax.xaxis.set_ticklabels(['Positive', 'Negative'])
            ax.yaxis.set_ticklabels(['Positive', 'Negative'])
            ## Display the visualization of the Confusion Matrix.
            fig1CF.savefig("fig1CF.png")
            image1CF = Image.open("fig1CF.png")
            st.image(image1CF)
            st.info(model1_cfMatrix)
            # explain confusion matrix
            with model1Report1_col2:
                st.write(
                'Confusion Matrix to show the prediction of results in the form of matrix.\n'
                '1. True Positive (TP): Correctly predicted.\n'
                '2. False Positive (FP): Incorrectly identified (Type I error).\n'
                '3. False Negative (FN): Incorrectly rejected (Type II error).\n'
                '4. True Negative (TN): Correctly rejected.\n'
                'In Model 1:  TP -> 42, FP -> 98, FN -> 6, TN -> 10. ')

                st.write("**Accuracy:**", (accuracy_score(y_test1, target_pred1).round(4)*100), "%")
                st.write("**Precision:**", (precision_score(y_test1, target_pred1).round(4) * 100), "%")
                st.write("**Recall:**", (recall_score(y_test1, target_pred1).round(4) * 100), "%")
    expand_Ada2 = st.expander(label='Model Evaluation: ROC Curve')
    with expand_Ada2:
        model1Report2_col1, model1Report2_col2 = st.columns(2)
        with model1Report2_col1:
            metrics.plot_roc_curve(adaboostclf, X_test1, y_test1)
            st.pyplot()
        # explain ROC
        with model1Report2_col2:
            st.write(
                'ROC Curve: shows the rating of model performance.\n'
                '1. The area covered by the curve is the area between the blue line (ROC) and the axis. \n'
                '2. This area covered is AUC.\n'
                '3. Ideal value for AUC is 1.\n'
                'Through this curve, it shows that Model 1 has a high performance.'
            )
    st.markdown("""<hr style="height:2px; border:none; color:#8CB1F5; background-color:#8CB1F5;" /> """,
                unsafe_allow_html=True)

    # display accuracy (SVM)
    annotated_text("", (" ⌨Model 2: Support Vector Machine (SVM)", "", "#88cbff"))
    st.write("\n")
    st.write(" **A. The Accuracy Result of Train Set**     ", (SVM_train_sc.round(4)*100),"%")
    st.write(" **B. The Accuracy Result of Test Set**     ", (SVM_test_sc.round(4)*100),"%")
    if SVM_test_sc > adaboostclf_test_sc and SVM_test_sc > ec_test_sc:
        st.success("**Conclusion** :Model 2 is the _**strongest**_ model!")
    elif SVM_test_sc < adaboostclf_test_sc and SVM_test_sc < ec_test_sc:
        st.error("**Conclusion** :Model 2 is the _**weakest**_ model!")
    elif SVM_test_sc > adaboostclf_test_sc and SVM_test_sc < ec_test_sc:
        st.warning("**Conclusion** :Model 2 is the _**stronger**_ than Model 1, but _**weaker**_ than Model 3!")
    elif SVM_test_sc < adaboostclf_test_sc and SVM_test_sc > ec_test_sc:
        st.warning("**Conclusion** :Model 2 is the _**stronger**_ than Model 3, but _**weaker**_ than Model 1!")
    else:
        st.error("**Conclusion** :Model 2 is the _**weakest**_ model!")
    expand_SVM1 = st.expander(label='Model Evaluation: Confusion Matrix')
    with expand_SVM1:
        model2Report1_col1, model2Report1_col2 = st.columns(2)
        with model2Report1_col1:
            # confusion matrix graph
            model2_cfMatrix = confusion_matrix(y_test3, target_pred2)
            fig2CF, ax = plt.subplots(figsize=(4, 4))
            ax = sns.heatmap(model2_cfMatrix, annot=True, cmap='YlGnBu')
            ax.set_title("Diabetes Prediction")
            ax.set_ylabel("Actual Value")
            ax.set_xlabel("Predicted Value")
            ## Ticket labels - List must be in alphabetical order
            ax.xaxis.set_ticklabels(['Positive', 'Negative'])
            ax.yaxis.set_ticklabels(['Positive', 'Negative'])
            ## Display the visualization of the Confusion Matrix.
            fig2CF.savefig("fig2CF.png")
            image2CF = Image.open("fig2CF.png")
            st.image(image2CF)
            st.info(model2_cfMatrix)
            with model2Report1_col2:
                st.write(
                    'Confusion Matrix to show the prediction of results in the form of matrix.\n'
                    '1. True Positive (TP): Correctly predicted.\n'
                    '2. False Positive (FP): Incorrectly identified (Type I error).\n'
                    '3. False Negative (FN): Incorrectly rejected (Type II error).\n'
                    '4. True Negative (TN): Correctly rejected. \n'
                    'In Model 2:  TP -> 51, FP -> 5, FN -> 7, TN -> 93. '
                )
    expand_SVM2 = st.expander(label='Model Evaluation: ROC Curve')
    with expand_SVM2:
        model2Report2_col1, model2Report2_col2 = st.columns(2)
        with model2Report2_col1:
            metrics.plot_roc_curve(SVM, X_test3, y_test3)
            st.pyplot()
        # explain ROC
        with model2Report2_col2:
            st.write(
                'ROC Curve: shows the rating of model performance.\n'
                '1. The area covered by the curve is the area between the blue line (ROC) and the axis. \n'
                '2. This area covered is AUC.\n'
                '3. Ideal value for AUC is 1.\n'
                'Through this curve, it shows that Model 1 has a high performance.'
            )
    st.markdown("""<hr style="height:2px; border:none; color:#8CB1F5; background-color:#8CB1F5;" /> """,
                unsafe_allow_html=True)

    # display accuracy (EM)
    annotated_text("", ("⌨Model 3: Ensemble Method", "", "#88cbff"))
    st.write("\n")
    st.write(" **A. The Accuracy Result of Train Set** ", (ec_train_sc.round(6)*100),"%")
    st.write(" **B. The Accuracy Result of Test Set** ", (ec_test_sc.round(6)*100),"%")
    if ec_test_sc > adaboostclf_test_sc and ec_test_sc > SVM_test_sc:
        st.success("**Conclusion** :Model 3 is the _**strongest**_ model!")
    elif ec_test_sc < adaboostclf_test_sc and ec_test_sc < SVM_test_sc:
        st.error("**Conclusion** :Model 3 is the _**weakest**_ model!")
    elif ec_test_sc > adaboostclf_test_sc and ec_test_sc < SVM_test_sc:
        st.warning("**Conclusion** :Model 3 is the _**stronger**_ than Model 1, but weaker than Model 2!")
    elif ec_test_sc < adaboostclf_test_sc and ec_test_sc > SVM_test_sc:
        st.warning("**Conclusion** :Model 3 is the _**stronger**_ than Model 2, but weaker than Model 1!")
    else:
        st.error("**Conclusion** :Model 3 is the _**weakest**_ model!")
    expand_EM1 = st.expander(label='Model Evaluation: Confusion Matrix')
    with expand_EM1:
        model3Report1_col1, model3Report1_col2 = st.columns(2)
        with model3Report1_col1:
            model3_cfMatrix = confusion_matrix(y_test2, target_pred3)
            fig3CF, ax = plt.subplots(figsize=(4, 4))
            ax = sns.heatmap(model3_cfMatrix, annot=True, cmap='YlGnBu')
            ax.set_title("Diabetes Prediction")
            ax.set_ylabel("Actual Value")
            ax.set_xlabel("Predicted Value")
            ## Ticket labels - List must be in alphabetical order
            ax.xaxis.set_ticklabels(['Positive', 'Negative'])
            ax.yaxis.set_ticklabels(['Positive', 'Negative'])
            ## Display the visualization of the Confusion Matrix.
            fig3CF.savefig("fig3CF.png")
            image3CF = Image.open("fig3CF.png")
            st.image(image3CF)
            st.info(model3_cfMatrix)
            with model3Report1_col2:
                st.write(
                    'Confusion Matrix to show the prediction of results in the form of matrix.\n'
                    '1. True Positive (TP): Correctly predicted.\n'
                    '2. False Positive (FP): Incorrectly identified (Type I error).\n'
                    '3. False Negative (FN): Incorrectly rejected (Type II error).\n'
                    '4. True Negative (TN): Correctly rejected.\n '
                    'In Model 3:  TP -> 52, FP -> 9, FN -> 10, TN -> 85. '
                )

                st.write("**Accuracy:**", (accuracy_score(y_test2, target_pred3).round(2) * 100), "%")
                st.write("**Precision:**", (precision_score(y_test2, target_pred3).round(4) * 100), "%")
                st.write("**Recall:**", (recall_score(y_test2, target_pred3).round(4) * 100), "%")
    st.markdown("""<hr style="height:5px; border:none; color:#594B44; background-color:#594B44;" /> """,
                unsafe_allow_html=True)
########################################################################################################################
# starting interface on title page
st.write("""# Diabetes Prediction Test""")
st.markdown("In this tool, you could compare the accuracy of 3 models in predicting diabetes results.")
st.markdown(" **👩‍💻Instructions to use this tool:** ")
st.markdown("1️⃣ Drag a diabetes dataset into the tool through side bar.")
annotated_text("2️⃣ Hit the", ("Generate Diabetes Prediction Result", "button", "#fea"), "to begin.")
st.write("\n")
st.markdown("3️⃣ You are now able to view the accuracy result of 3 models.")

# upload file through sidebar
st.sidebar.title('Upload your diabetes CSV data here')
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
if uploaded_file is not None:
    with st.spinner('CSV file is uploading...'):
        time.sleep(3)
    st.success('CSV file successfully uploaded.')
else:
    st.error('Awaiting for CSV file to be uploaded.')

# upload file through sidebar
if uploaded_file is not None:
    DF = pd.read_csv(uploaded_file)
    st.subheader('Section 1: Dataset')
    st.markdown('**Glimpse of dataset**')
    st.write(DF)
    if st.button("Generate Diabetes Prediction Result"):
        st.markdown("""<hr style="height:5px; border:none; color:#594B44; background-color:#594B44;" /> """,
                    unsafe_allow_html=True)
        build_model(DF)
########################################################################################################################

