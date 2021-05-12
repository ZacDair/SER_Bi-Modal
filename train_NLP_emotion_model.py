# BET EMOTIONS [Fear, Anger, Joy, Sadness, Disgust, Surprise]
import pandas as pd
import sklearn
from keras.utils import np_utils
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, multilabel_confusion_matrix, classification_report
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from nltk.corpus import stopwords
import os
import pickle
from sklearn.pipeline import Pipeline


# Save the model and weights
def saveModel(modelName, model):
    save_dir = os.path.join(os.getcwd(), 'my_models')
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    model_path = os.path.join(save_dir, modelName+".sav")
    pickle.dump(model, open(model_path, 'wb'))
    print('Saved trained model at %s ' % model_path)


# Load the model
def loadModel(filepath):
    loaded_model = pickle.load(open(filepath, 'rb'))
    return loaded_model


# Change the plot styling
plt.style.use('ggplot')


# Plot the accuracy vs val accuracy and loss vs val loss
def plot_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()


# Read in the emotional text dataset
data = pd.read_csv("..\\Datasets\\ISEARs.csv", sep='|', usecols=["SIT", "Field1"]) # Emotion Range: [joy, fear, anger, sadness, disgust, shame, guilt]
# Remove \n chars
data = data.rename(columns={"SIT": "text", "Field1":"label"})
data["text"] = data["text"].str.replace("\n", "")


testableData = data[data["label"] != "shame"]
testableData = testableData[testableData["label"] != "guilt"]
testableData['text'] = testableData['text'].str.lower()

print("Class Distribution:")
print(testableData["label"].value_counts())
print(len(data))
# Convert the labels to numbers
# print(testableData["label"].unique())
# testableData["label"] = testableData["label"].replace("joy", 1)
# testableData["label"] = testableData["label"].replace("fear", 2)
# testableData["label"] = testableData["label"].replace("anger", 3)
# testableData["label"] = testableData["label"].replace("sadness", 4)
# testableData["label"] = testableData["label"].replace("disgust", 5)

sentences = testableData["text"].values
y = testableData['label'].values

sentences_train, sentences_test, y_train, y_test = train_test_split(sentences, y, test_size=0.25, random_state=1000)

lb = LabelEncoder()
y_train_og = y_train
y_test_og = y_test
y_train = np_utils.to_categorical(lb.fit_transform(y_train))
y_test = np_utils.to_categorical(lb.fit_transform(y_test))

# Vectorizer Definition
vectorizer = CountVectorizer(min_df=0, lowercase=True)
vectorizer.fit(sentences_train)
X_train = vectorizer.transform(sentences_train)
X_test = vectorizer.transform(sentences_test)

'''Our MODEL'''
# Logistic Regression - Baseline 65%
classifier = LogisticRegression(max_iter=1000)
classifier.fit(X_train, y_train_og)
preds = classifier.predict(X_test)
print(len(preds))
print("LogReg Accuracy:", accuracy_score(y_test_og, preds))
print("Confusion Matrix:")
print(multilabel_confusion_matrix(y_test_og, preds))
print("Report:")
print(classification_report(y_test_og, preds))

'''Their BASELINE MODEL'''
# Read in the emotional text dataset
data = pd.read_csv("..\\Datasets\\RAVDESS\\utterancesFull.csv") # Emotion Range: [joy, fear, anger, sadness, disgust, shame, guilt]
data["text"] = data["text"].str.replace("\n", "")

# drop unused emotions
#testableData = data[data["label"] != "shame"]
#testableData = testableData[testableData["label"] != "guilt"]
testableData = data
# print class distibution
print("BASELINE Class Distribution:")
print(testableData["label"].value_counts())

# Remove stop words from the sentences
stop_words = set(stopwords.words('english'))
testableData['text'] = testableData['text'].str.lower().str.split()
testableData['text'] = testableData['text'].apply(lambda x: [item for item in x if item not in stop_words])
testableData['text'] = testableData['text'].str.join(" ")
sentences = testableData["text"].values

y = testableData['label'].values

sentences_train, sentences_test, y_train, y_test = train_test_split(sentences, y, test_size=0.25, random_state=1000)
lb = LabelEncoder()
y_train_og = y_train
y_test_og = y_test


text_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', LogisticRegression(max_iter=1000))])
classifier = text_clf
classifier.fit(sentences_train, y_train_og)
preds = classifier.predict(sentences_test)
print("LogReg Accuracy:", accuracy_score(y_test_og, preds))
print("Confusion Matrix:")
print(multilabel_confusion_matrix(y_test_og, preds))
print("Report:")
print(classification_report(y_test_og, preds))