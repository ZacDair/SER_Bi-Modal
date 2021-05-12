# BET EMOTIONS [Fear, Anger, Joy, Sadness, Disgust, Surprise]
import pandas as pd
from keras.utils import np_utils
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.cluster import KMeans
from sklearn import svm
from sklearn.tree import ExtraTreeClassifier
from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from keras.backend import clear_session
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder


# Changes Pandas display options to support the large strings
pd.options.display.max_colwidth = 200

# Read in the emotional text dataset
data = pd.read_csv("..\\Datasets\\ISEAR.csv") # Emotion Range: [joy, fear, anger, sadness, disgust, shame, guilt]

# Remove \n chars
data["text"] = data["text"].str.replace("\n", "")


# Drop emotions we don't use
testableData = data[data["label"] != "shame"]
testableData = testableData[testableData["label"] != "guilt"]

# Convert the labels to numbers
# print(testableData["label"].unique())
#testableData["label"] = testableData["label"].replace("joy", 1)
#testableData["label"] = testableData["label"].replace("fear", 2)
#testableData["label"] = testableData["label"].replace("anger", 3)
#testableData["label"] = testableData["label"].replace("sadness", 4)
#testableData["label"] = testableData["label"].replace("disgust", 5)

sentences = testableData["text"].values
y = testableData['label'].values

sentences_train, sentences_test, y_train, y_test = train_test_split(sentences, y, test_size=0.25, random_state=1000)

lb = LabelEncoder()
y_train_og = y_train
y_test_og = y_test
y_train = np_utils.to_categorical(lb.fit_transform(y_train))
y_test = np_utils.to_categorical(lb.fit_transform(y_test))

# Vectorizer Definition
vectorizer = CountVectorizer(min_df=0, lowercase=False)
vectorizer.fit(sentences_train)
X_train = vectorizer.transform(sentences_train)
X_test = vectorizer.transform(sentences_test)

'''BASELINE MODELS'''
# Logistic Regression - Baseline 65%
classifier = LogisticRegression(max_iter=1000)
classifier.fit(X_train, y_train_og)
preds = classifier.predict(X_test)
print("LogReg Accuracy:", accuracy_score(y_test_og, preds))
# print("Confusion Matrix:")
# print(multilabel_confusion_matrix(y_test_og, preds))
clear_session()

# Perceptron - 62% or 63% when using numbers instead of strings for labels
classifier = Perceptron()
classifier.fit(X_train, y_train_og)
preds = classifier.predict(X_test)
print("Perceptron Accuracy:", accuracy_score(y_test_og, preds))
# print("Confusion Matrix:")
# print(multilabel_confusion_matrix(y_test_og, preds))
clear_session()

# Random Forest -
classifier = RandomForestClassifier()
classifier.fit(X_train, y_train_og)
preds = classifier.predict(X_test)
print("Random Forest Accuracy:", accuracy_score(y_test_og, preds))
# print("Confusion Matrix:")
# print(multilabel_confusion_matrix(y_test_og, preds))
clear_session()

# SGD Classifier -
classifier = SGDClassifier()
classifier.fit(X_train, y_train_og)
preds = classifier.predict(X_test)
print("SGDClassifier Accuracy:", accuracy_score(y_test_og, preds))
# print("Confusion Matrix:")
# print(multilabel_confusion_matrix(y_test_og, preds))
clear_session()

# Dummy  -
classifier = DummyClassifier()
classifier.fit(X_train, y_train_og)
preds = classifier.predict(X_test)
print("Dummy Accuracy:", accuracy_score(y_test_og, preds))
# print("Confusion Matrix:")
# print(multilabel_confusion_matrix(y_test_og, preds))
clear_session()

# AdaBoostClassifier  -
classifier = AdaBoostClassifier()
classifier.fit(X_train, y_train_og)
preds = classifier.predict(X_test)
print("AdaBoostClassifier Accuracy:", accuracy_score(y_test_og, preds))
# print("Confusion Matrix:")
# print(multilabel_confusion_matrix(y_test_og, preds))
clear_session()

#  KNeighborsClassifier  -
classifier = KNeighborsClassifier()
classifier.fit(X_train, y_train_og)
preds = classifier.predict(X_test)
print("KNeighborsClassifier Accuracy:", accuracy_score(y_test_og, preds))
# print("Confusion Matrix:")
# print(multilabel_confusion_matrix(y_test_og, preds))
clear_session()


# SVM  -
classifier = svm.SVC()
classifier.fit(X_train, y_train_og)
preds = classifier.predict(X_test)
print("SVC Accuracy:", accuracy_score(y_test_og, preds))
# print("Confusion Matrix:")
# print(multilabel_confusion_matrix(y_test_og, preds))
clear_session()

# SVM  -
classifier = svm.LinearSVC()
classifier.fit(X_train, y_train_og)
preds = classifier.predict(X_test)
print("LinearSVC Accuracy:", accuracy_score(y_test_og, preds))
# print("Confusion Matrix:")
# print(multilabel_confusion_matrix(y_test_og, preds))
clear_session()

# Extra Tree  -
classifier = ExtraTreeClassifier()
classifier.fit(X_train, y_train_og)
preds = classifier.predict(X_test)
print("ExtraTreeClassifier Accuracy:", accuracy_score(y_test_og, preds))
# print("Confusion Matrix:")
# print(multilabel_confusion_matrix(y_test_og, preds))
clear_session()
