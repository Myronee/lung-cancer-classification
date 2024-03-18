import numpy as pd
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier


st.header('Lung Cancer Classification')


st.sidebar.header('Specify input parameters')


def user_input_featuers():
    Gender = st.sidebar.slider('Male/Female', 0, 1)
    Age = st.sidebar.slider('Age', 20, 90)
    Smoking = st.sidebar.slider('Smoking', 0, 1)
    Yellow = st.sidebar.slider('Yellow', 0, 1)
    Anxiety = st.sidebar.slider('Anxiety', 0, 1)
    Peer_pressure = st.sidebar.slider('Peer_pressure', 0, 1)
    Chronic_Disease = st.sidebar.slider('Chronic_Disease', 0, 1)
    Fatigue = st.sidebar.slider('Fatigue', 0, 1)
    Allergy = st.sidebar.slider('Allergy', 0, 1)
    Wheezing = st.sidebar.slider('Wheezing', 0, 1)
    Alcohol = st.sidebar.slider('Alcohol', 0, 1)
    Coughing = st.sidebar.slider('Coughing', 0, 1)
    Shortness_of_Breath = st.sidebar.slider('Shortness_of_Breath', 0, 1)
    Swallowing_Difficulty = st.sidebar.slider('Swallowing_Difficulty', 0, 1)
    Chest_pain = st.sidebar.slider('Chest_pain', 0, 1)

    data = {
        'GENDER': Gender,
        'AGE': Age,
        'SMOKING': Smoking,
        'YELLOW_FINGERS': Yellow,
        'ANXIETY': Anxiety,
        'PEER_PRESSURE': Peer_pressure,
        'CHRONIC DISEASE': Chronic_Disease,
        'FATIGUE ': Fatigue,
        'ALLERGY ': Allergy,
        'WHEEZING': Wheezing,
        'ALCOHOL CONSUMING': Alcohol,
        'COUGHING': Coughing,
        'SHORTNESS OF BREATH': Shortness_of_Breath,
        'SWALLOWING DIFFICULTY': Swallowing_Difficulty,
        'CHEST PAIN': Chest_pain,
    }

    features = pd.DataFrame(data, index=[0])
    return features


def add_parameter_ui(clf_name):
    params = dict()
    if clf_name == "Random Forest":
        max_depth = st.slider("max_depth", 2, 15)
        n_estimators = st.slider("number of estimators", 1, 100)
        params["max_depth"] = max_depth
        params["n_estimators"] = n_estimators

    elif clf_name == "KNN":
        k = st.slider("K", 1, 15)
        params["k"] = k
    else:
        c = st.slider("c", 0.01, 10.0)
        params["c"] = c

    return params


def get_classifier(clf_name, params):

    if clf_name == "KNN":
        clf = KNeighborsClassifier(n_neighbors=params["k"])
    elif clf_name == "Logistic Regression":
        clf = LogisticRegression(C=params["c"])
    else:
        clf = RandomForestClassifier(n_estimators=params["n_estimators"], max_depth=params["max_depth"],
                                     random_state=1234)

    return clf


classifier_name = st.selectbox(
    "Select Classifier", ("Random Forest", "KNN", "Logistic Regression"))

df_input = user_input_featuers()
params = add_parameter_ui(classifier_name)
clf = get_classifier(classifier_name, params)

st.header('Specified input parameters')
st.write(df_input)

# fix data and buid model

df = pd.read_csv('../data_folder/survey_lung_cancer.csv')
df.drop_duplicates(inplace=True)
y = df['LUNG_CANCER']

le = LabelEncoder()
df['LUNG_CANCER'] = le.fit_transform(df['LUNG_CANCER'])
df['GENDER'] = le.fit_transform(df['GENDER'])

df_input['GENDER'] = le.fit_transform(df_input['GENDER'])
X_untouched = df.drop(['LUNG_CANCER'], axis=1)

X = df.drop(['LUNG_CANCER'], axis=1)
y = df['LUNG_CANCER']

for i in X.columns[2:]:
    holder = []
    for idx, j in enumerate(df[i]):
        holder.append(j-1)
    X[i] = holder

X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=11, stratify=y, test_size=0.2)

sc = StandardScaler()
X_train['AGE'] = sc.fit_transform(X_train[['AGE']])
X_test['AGE'] = sc.transform(X_test[['AGE']])


clf.fit(X_train, y_train)
y_pred_input = clf.predict(df_input)
y_pred_input_prob = clf.predict_proba(df_input)
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
st.write(f"Classifier: {clf}")
st.write(f"Accuracy on Test Data: {acc}")

st.header('Lung cancer prediction from user input')
st.write(y_pred_input_prob)
st.write(y_pred_input)
st.write(
    f"Prediction: {round((y_pred_input_prob[0][1]*100),3)} % risk of lung cancer")
st.set_option('deprecation.showPyplotGlobalUse', False)

if classifier_name == 'Random Forest':
    rf = RandomForestClassifier()
    rf.fit(X_untouched, y)
    if st.button('Show Feature Importance Plot'):
        st.header('Fauture importance')
        fig, ax = plt.subplots()
        plt.title('Feuture Importance')
        importance = rf.feature_importances_
        feature_names = [X_untouched.columns[i]
                         for i in range(X_untouched.shape[1])]
        forest_importance = pd.Series(importance, index=feature_names)
        forest_importance.plot.barh()
        st.pyplot(fig)
