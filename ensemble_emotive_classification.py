import joblib
import pandas as pd
from keras.utils import np_utils
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, multilabel_confusion_matrix, classification_report
from sklearn.model_selection import KFold
from sklearn import metrics
import numpy as np
from keras.layers import Dense
from keras.layers import Conv1D
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Activation, MaxPooling1D
from keras.models import Sequential
from keras.backend import clear_session
from sklearn.pipeline import Pipeline


def createCNN():
    model = Sequential()
    model.add(Conv1D(128, 5, padding='same',
                     input_shape=(53, 1)))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(MaxPooling1D(pool_size=(8)))
    model.add(Conv1D(128, 5, padding='same'))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(8))
    model.add(Activation('softmax'))

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    return model


# Fit the model to our data (returns the model history)
def model_fit_CNN(model, x_train, y_train, batchSize, epochs, x_test, y_test):
    # Changing Dimension for CNN model
    x_train = np.expand_dims(x_train, axis=2)
    x_test = np.expand_dims(x_test, axis=2)
    cnnhistory = model.fit(x_train, y_train, batch_size=batchSize, epochs=epochs, validation_data=(x_test, y_test), verbose=False)
    return cnnhistory


# Train, Test the model, plot the accuracy results
def runKFoldAPModel(featureData, labelData):
    # K-fold init
    kf = KFold(n_splits=5, shuffle=True)
    iteration = 0
    totalAcc = 0
    bestAcc = 0
    bestModel = ""
    mX = []
    mY = []
    for trainIndex, testIndex, in kf.split(featureData):

        xTrain = featureData[trainIndex]
        yTrain = labelData[trainIndex]
        xTest = featureData[testIndex]
        yTest = labelData[testIndex]

        AP_Model = createCNN()

        model_fit_CNN(AP_Model, xTrain, yTrain, 16, 1000, xTest, yTest)

        pred = AP_Model.predict(xTest)

        mX.extend(pred)
        mY.extend(yTest)

        print("\nPredictions:\n", pred)
        score = metrics.accuracy_score(yTest, pred)
        if score > bestAcc:
            bestModel = AP_Model
            bestAcc = score
        totalAcc += score
        print("Model Accuracy:", score)
        print("Conf Matrix:\n", metrics.confusion_matrix(yTest, pred))
        print(metrics.classification_report(yTest, pred))

        clear_session()
        iteration += 1
    print("Avg Accuracy:", totalAcc/5)
    return bestModel, mX, np.array(mY)


# Train, Test the combined model system
def runKFoldCombinedModel(featureData, labelData, textData):
    # K-fold init
    kf = KFold(n_splits=5, shuffle=True)
    iteration = 0
    totalAccA = 0
    bestAccA = 0
    bestModelA = ""
    totalAccT = 0
    bestAccT = 0
    bestModelT = ""
    mX = []
    mY = []

    for trainIndex, testIndex, in kf.split(featureData):

        xTrainA = featureData[trainIndex]
        xTestA = featureData[testIndex]
        xTestA = np.expand_dims(xTestA, axis=2)

        yTrain = labelData[trainIndex]
        yTest = labelData[testIndex]

        xTrainT = textData.iloc[trainIndex]
        xTestT = textData.iloc[testIndex]

        # Create, fit and predict with the CNN
        AP_Model = createCNN()
        model_fit_CNN(AP_Model, xTrainA, yTrain, 16, 300, xTestA, yTest)
        predA = AP_Model.predict(xTestA)

        # Display Model Score
        scoreA = metrics.accuracy_score(yTest, predA.argmax(axis=1))
        # print("CNN Accuracy:", scoreA)
        # print("Conf Matrix:\n", metrics.confusion_matrix(yTest, predA.argmax(axis=1)))
        # print(metrics.classification_report(yTest, predA.argmax(axis=1)))

        # Create, fit and predict with the Log Reg
        text_clf = Pipeline(
            [('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', LogisticRegression(max_iter=1000))])
        text_clf.fit(xTrainT, yTrain)

        # Display Model Score
        predT = text_clf.predict(xTestT)
        scoreT = accuracy_score(yTest, predT)
        # print("LogReg Accuracy:", scoreT)
        # print("Confusion Matrix:")
        # print(multilabel_confusion_matrix(yTest, predT))
        # print("Report:")
        # print(classification_report(yTest, predT))

        mergedPreds = np.column_stack((predA, predT))
        mX.extend(mergedPreds)
        mY.extend(yTest)

        if scoreA > bestAccA:
            bestModelA = AP_Model
            bestAccA = scoreA
        totalAccA += scoreA

        if scoreT > bestAccT:
            bestModelT = text_clf
            bestAccT = scoreT
        totalAccT += scoreT
        clear_session()
        iteration += 1

    print("Avg CNN Accuracy:", totalAccA / 5)
    print("Avg LogReg Accuracy:", totalAccT / 5)
    return bestModelA, bestModelT, mX, np.array(mY)


X = joblib.load("E:\\FYP-Implementation\\Emotion-Classification-Ravdess\\joblib_features\\X-SAVEE.joblib")
y = joblib.load("E:\\FYP-Implementation\\Emotion-Classification-Ravdess\\joblib_features\\y-SAVEE.joblib")
textdf = pd.read_csv("../Datasets/SAVEE/utterancesFull.csv")
# Remove stop words from the sentences
stop_words = set(stopwords.words('english'))
textdf['text'] = textdf['text'].str.lower().str.split()
textdf['text'] = textdf['text'].apply(lambda x: [item for item in x if item not in stop_words])
textdf['text'] = textdf['text'].str.join(" ")
sentences = textdf["text"]

kf = KFold(n_splits=5, shuffle=True)
totalAcc = 0
bestAcc = 0
# META Train Test Split
for metaTrainIndex, metaTestIndex, in kf.split(X):
    metaValX = X[metaTestIndex]
    metaValText = sentences.iloc[metaTestIndex]
    metaValY = y[metaTestIndex]

    best_Audio_Model, best_Text_Model, metaTrainX, metaTrainY = runKFoldCombinedModel(X[metaTrainIndex], y[metaTrainIndex], sentences[metaTrainIndex])

    metaModel = LogisticRegression(max_iter=1000)
    metaModel.fit(metaTrainX, metaTrainY)

    metaValX = np.expand_dims(metaValX, axis=2)
    predsA = best_Audio_Model.predict(metaValX)
    predsT = best_Text_Model.predict(metaValText)
    print("Sub Model Scores:", metrics.accuracy_score(metaValY, predsA.argmax(axis=1)), metrics.accuracy_score(metaValY, predsT))
    mergedPreds = np.column_stack((predsA, predsT))
    meta_preds = metaModel.predict(mergedPreds)
    score = metrics.accuracy_score(metaValY, meta_preds)
    print(classification_report(metaValY, meta_preds))
    if score > bestAcc:
        bestAcc = score
    totalAcc += score
    print("Meta Model Score:", score)
print("Meta Model Avg Score:", totalAcc/5)
print("Meta Model Best Score:", bestAcc)


'''
Generalisation Experiment
Second Running of the Ensemble at this stage it's trained on the previous dataset
'''
# print("Generalisation Experiment")
# # Load SAVEE Data
# X = joblib.load("E:\\FYP-Implementation\\Emotion-Classification-Ravdess\\joblib_features\\X-RAV.joblib")
# y = joblib.load("E:\\FYP-Implementation\\Emotion-Classification-Ravdess\\joblib_features\\y-RAV.joblib")
# textdf = pd.read_csv("../Datasets/RAVDESS/utterancesFull.csv")
# # Remove stop words from the sentences
# stop_words = set(stopwords.words('english'))
# textdf['text'] = textdf['text'].str.lower().str.split()
# textdf['text'] = textdf['text'].apply(lambda x: [item for item in x if item not in stop_words])
# textdf['text'] = textdf['text'].str.join(" ")
# sentences = textdf["text"]
#
# metaValX = X
# metaValText = sentences
# metaValY = y
#
# metaValX = np.expand_dims(metaValX, axis=2)
# predsA = best_Audio_Model.predict(metaValX)
# predsT = best_Text_Model.predict(metaValText)
# print("Sub Model Scores:", metrics.accuracy_score(metaValY, predsA.argmax(axis=1)),
#       metrics.accuracy_score(metaValY, predsT))
# mergedPreds = np.column_stack((predsA, predsT))
# meta_preds = metaModel.predict(mergedPreds)
# score = metrics.accuracy_score(metaValY, meta_preds)
# print(classification_report(metaValY, meta_preds))