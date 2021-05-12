import os.path
import moviepy.editor as mp
import numpy as np
import librosa
import pandas as pd
import os
from jiwer import wer
from keras.utils.vis_utils import plot_model
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import optimizers
import model_io_functions
from sklearn.utils import shuffle
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.metrics import accuracy_score, multilabel_confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten, Dropout, Activation
from keras.layers import Conv1D, MaxPooling1D
import matplotlib.pyplot as plt
import time
from keras.backend import clear_session
from sklearn import model_selection, linear_model, metrics, svm
import speech_recognition as sr

modelHistoryData = pd.DataFrame(columns=["loss", "accuracy", "val_loss", "val_accuracy", "date", "model", "train_length", "test_length"])


# Plot the CNN History of a model (train/test accuracy and train/test loss)
def plot_history(titleDetail, history, saveImgPath):
    plt.style.use('ggplot')

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)
    plt.figure(figsize=(10, 5))
    plt.plot(x, acc, 'royalblue', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    #plt.title('Training and validation accuracy' + " " + titleDetail)
    plt.legend()
    plt.plot(x, loss, 'lightsteelblue', label='Training loss')
    plt.plot(x, val_loss, 'rosybrown', label='Validation loss')
    plt.title('CNN Train/Test History - ' + " " + titleDetail)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy/Loss")
    plt.legend()

    t = time.time()
    plt.savefig(saveImgPath+"CNN_History_"+str(t)+".png")

    plt.show()


# Transcribe a WAV file to text and return the result
def transcribe(audioFilePath):
    try:
        audioFile = sr.AudioFile(audioFilePath)

        # Define the speech recognising method (Google, IBM..etc)
        r = sr.Recognizer()

        # Use the audio file as the source
        with audioFile as source:
            audio = r.record(source)  # read the entire audio file

        return r.recognize_google(audio)
    except sr.UnknownValueError:
        return "failed transcription"


# Calculate the word error rate
def calculateWER(transcribed, actual):
    return wer(actual, transcribed)


# Translate the audio to text based on EMO_DB filename labels
def translate_EMODB_file(audioFilePath):
    item = os.path.basename(audioFilePath)
    if item[2:5] == "a01":
        content = "The tablecloth is lying on the fridge"
    elif item[2:5] == "a02":
        content = "She wants to hand that in on Wednesday"
    elif item[2:5] == "a04":
        content = "I could tell him tonight"
    elif item[2:5] == "a05":
        content = "The black piece of paper is up there next to the piece of wood"
    elif item[2:5] == "a07":
        content = "It will be ready in seven hours"
    elif item[2:5] == "b01":
        content = "What kind of bags are there under the table"
    elif item[2:5] == "b02":
        content = "They just carried it up and now they're going down again"
    elif item[2:5] == "b03":
        content = "I used to go home on the weekends and visit Agnes"
    elif item[2:5] == "b09":
        content = "I want to take that away and then go have a drink with Karl"
    elif item[2:5] == "b10":
        content = "It will be in the place where we always put it."
    else:
        print("unknown translation:", item)
        content = "unknown"
    return content


def getSAVEEText(audioFilePath):
    item = os.path.basename(audioFilePath)


    return

def returnFileInfoRAVDESS(file):
    emotion, fileList = labelRAVDESS([file])
    name = os.path.basename(file)
    # Utilise the naming convention to discern the spoken content (or labels)
    nameSegments = name.split("-")
    if nameSegments[4] == "01":
        statement = "kids are talking by the door"
    elif nameSegments[4] == "02":
        statement = "dogs are sitting by the door"
    else:
        statement = "UNSURE"
        print("Unsure what the statement was")
    if len(emotion) == 0:
        emotion = "Not Found - Unsupported Emotion"
        return emotion, statement
    return emotion.values, statement


def returnFileInfoSAVEE(file):
    emotion, fileList = label_SAVEE([file])
    if len(emotion) == 0:
        emotion = "Not Found - Unsupported Emotion"
        return emotion
    return emotion.values


'''
*------------------------------------------------*
|                                                |
|   Affective Prosody (Audio) Model Functions    |
|                                                |
*------------------------------------------------*
'''


# Get all audio files in dirs and sub dirs
def getAllMP3Files(pathToSearch):
    filePathList = []
    for root, dirs, files in os.walk(pathToSearch):
        for file in files:
            # append the file name to the list
            if file.endswith(".mp3"):
                filePathList.append(os.path.join(root, file))

    print("Found", len(filePathList), "mp3 files in", pathToSearch)
    return filePathList


# Get all audio files in dirs and sub dirs
def getAllWavFiles(pathToSearch):
    filePathList = []
    for root, dirs, files in os.walk(pathToSearch):
        for file in files:
            # append the file name to the list
            if file.endswith(".wav"):
                filePathList.append(os.path.join(root, file))

    print("Found", len(filePathList), "WAV files in", pathToSearch)
    return filePathList


# # Setting Labels based on filenames - RAVDESS Labelling
# def labelRAVDESS(files):
#     feeling_list = []
#     labelledFiles = []
#     utterances = []
#     for item in files:
#         itemPath = item
#         item = os.path.basename(item)
#         currentLength = len(feeling_list)
#         if item[6:-16] == '02' and int(item[18:-4]) % 2 == 0:
#             feeling_list.append('female_calm')
#
#         elif item[6:-16] == '02' and int(item[18:-4]) % 2 == 1:
#             feeling_list.append('male_calm')
#         elif item[6:-16] == '03' and int(item[18:-4]) % 2 == 0:
#             feeling_list.append('female_joy')
#
#         elif item[6:-16] == '03' and int(item[18:-4]) % 2 == 1:
#             feeling_list.append('male_joy')
#         elif item[6:-16] == '04' and int(item[18:-4]) % 2 == 0:
#             feeling_list.append('female_sad')
#
#         elif item[6:-16] == '04' and int(item[18:-4]) % 2 == 1:
#             feeling_list.append('male_sad')
#
#         elif item[6:-16] == '05' and int(item[18:-4]) % 2 == 0:
#             feeling_list.append('female_anger')
#
#         elif item[6:-16] == '05' and int(item[18:-4]) % 2 == 1:
#             feeling_list.append('male_anger')
#
#         elif item[6:-16] == '06' and int(item[18:-4]) % 2 == 0:
#             feeling_list.append('female_fear')
#
#         elif item[6:-16] == '06' and int(item[18:-4]) % 2 == 1:
#             feeling_list.append('male_fear')
#         elif item[:1] == 'a':
#             feeling_list.append('male_anger')
#         elif item[:1] == 'f':
#             feeling_list.append('male_fear')
#         elif item[:1] == 'h':
#             feeling_list.append('male_joy')
#         # elif item[:1]=='n':
#         # feeling_list.append('neutral')
#         elif item[:2] == 'sa':
#             feeling_list.append('male_sad')
#
#         if currentLength < len(feeling_list):
#             nameSegments = item.split("-")
#             if nameSegments[4] == "01":
#                 statement = "kids are talking by the door"
#             elif nameSegments[4] == "02":
#                 statement = "dogs are sitting by the door"
#             else:
#                 statement = "UNSURE"
#             utterances.append(statement)
#             labelledFiles.append(itemPath)
#
#     # Store the labels in a DataFrame made from the above feeling list
#     labels = pd.DataFrame(feeling_list)
#     utter = pd.DataFrame(utterances, columns=["feature"])
#     df1 = createFeatureLabelDataFrame(utter, labels)
#     df1.to_csv("../Datasets/RAVDESS/utterancesFull.csv", index=False)
#
#     # Print the length of both our labels and our usable files
#     print(len(feeling_list), "Labels")
#     print(len(labelledFiles), "Usable Files")
#
#     return labels, labelledFiles

def labelRAVDESS(files):
    feeling_list = []
    labelledFiles = []
    utterances = []
    for item in files:
        itemPath = item
        item = os.path.basename(item)
        currentLength = len(feeling_list)
        file = int(item[7:8]) - 1 # RAVDESS
        feeling_list.append(file)

        if currentLength < len(feeling_list):
            nameSegments = item.split("-")
            if nameSegments[4] == "01":
                statement = "kids are talking by the door"
            elif nameSegments[4] == "02":
                statement = "dogs are sitting by the door"
            else:
                statement = "UNSURE"
            utterances.append(statement)
            labelledFiles.append(itemPath)

    # Store the labels in a DataFrame made from the above feeling list
    labels = pd.DataFrame(feeling_list)
    utter = pd.DataFrame(utterances, columns=["feature"])
    df1 = createFeatureLabelDataFrame(utter, labels)
    df1.to_csv("../Datasets/RAVDESS/utterancesFull.csv", index=False)

    # Print the length of both our labels and our usable files
    print(len(feeling_list), "Labels")
    print(len(labelledFiles), "Usable Files")

    return labels, labelledFiles


# Setting Labels based on filenames - EMO_DB Labelling
def label_EMO_DB(files):
    feeling_list = []
    labelledFiles = []
    utterances = []

    # List of ID's that correspond to male actors
    maleIds = ["03", "10", "11", "12", "15"]
    femaleIDs = ["08", "09", "13", "14", "16"]
    for item in files:
        itemPath = item
        item = os.path.basename(item)
        currentLength = len(feeling_list)

        # Discern male or female actor
        if item[0:2] in maleIds:
            gender = "male"
        elif item[0:2] in femaleIDs:
            gender = "female"
        else:
            print("unknown gender:", item)
            gender = "unknown"

        # Discern the exhibited emotion
        if item[5] == "W":
            feeling_list.append(gender+"_anger")
        elif item[5] == "L":
            feeling_list.append(gender+"_boredom")
        elif item[5] == "E":
            feeling_list.append(gender+"_disgust")
        elif item[5] == "A":
            feeling_list.append(gender+"_fear")
        elif item[5] == "F":
            feeling_list.append(gender+"_joy")
        elif item[5] == "T":
            feeling_list.append(gender+"_sad")
        elif item[5] == "N":
            feeling_list.append(gender + "_neutral")
        else:
            print("Emotion labelling issue with file:", item)

        if currentLength < len(feeling_list):
            utterances.append(translate_EMODB_file(itemPath))
            labelledFiles.append(itemPath)

    # Store the labels in a DataFrame made from the above feeling list
    labels = pd.DataFrame(feeling_list)
    utter = pd.DataFrame(utterances, columns=["feature"])
    df1 = createFeatureLabelDataFrame(utter, labels)
    df1.to_csv("../Datasets/EMO_DB/utterancesFull.csv", index=False)
    # Print the length of both our labels and our usable files
    print(len(feeling_list), "Labels")
    print(len(labelledFiles), "Usable Files")

    return labels, labelledFiles


# Setting Labels based on filenames - EMO_DB Labelling
def label_SAVEE(files):
    feeling_list = []
    labelledFiles = []
    utterances = []

    for item in files:
        itemPath = item
        item = os.path.basename(item)
        currentLength = len(feeling_list)
        # Discern the exhibited emotion 'a', 'd', 'f', 'h', 'n', 'sa' and 'su' (anger, disust, fear, happy, neutral, sad, surprised)
        if item[0] == "a":
            feeling_list.append(0)
        elif item[0] == "d":
            feeling_list.append(1)
        elif item[0] == "f":
            feeling_list.append(2)
        elif item[0] == "h":
            feeling_list.append(3)
        elif item[0] == "n":
            feeling_list.append(4)
        elif item[0:2] == "sa":
            feeling_list.append(5)
        elif item[0:2] == "su":
            feeling_list.append(6)
        else:
            print("Emotion labelling issue with file:", item)

        if currentLength < len(feeling_list):
            content = transcribe(itemPath)
            utterances.append(content)
            labelledFiles.append(itemPath)

    # Store the labels in a DataFrame made from the above feeling list
    labels = pd.DataFrame(feeling_list)
    utter = pd.DataFrame(utterances, columns=["feature"])
    df1 = createFeatureLabelDataFrame(utter, labels)
    df1.to_csv("../Datasets/SAVEE/utterancesFull.csv", index=False)

    # Print the length of both our labels and our usable files
    print(len(feeling_list), "Labels")
    print(len(labelledFiles), "Usable Files")

    return labels, labelledFiles


# Setting Labels based on filenames - MELD Labelling
def label_MELD(files):
    feeling_list = []
    utterances = []
    labelledFiles = []
    dfDev = pd.read_csv("D:/MELD.Raw.tar/MELD.Raw/dev_sent_emo.csv", usecols=["Emotion","Sentiment","Dialogue_ID","Utterance_ID", "Utterance"])
    dfTrain = pd.read_csv("D:/MELD.Raw.tar/MELD.Raw/train.tar/train/train_sent_emo.csv",usecols=["Emotion", "Sentiment", "Dialogue_ID", "Utterance_ID", "Utterance"])
    emotionCount = {}
    for item in files:
        itemPath = item
        if itemPath.find("MELD-Train") != -1:
            df = dfTrain
        else:
            df = dfDev

        item = os.path.basename(item)
        currentLength = len(feeling_list)

        temp = item.split("_")
        dialogueID = str(temp[0]).replace('dia', '')
        utteranceID = str(temp[1]).replace('utt', '')
        utteranceID = utteranceID.replace('.mp3', '')
        emotion = df[(df["Dialogue_ID"] == int(dialogueID)) & (df["Utterance_ID"] == int(utteranceID))]["Emotion"]
        utt = df[(df["Dialogue_ID"] == int(dialogueID)) & (df["Utterance_ID"] == int(utteranceID))]["Utterance"]
        if len(emotion.values) != 0:
            if str(emotion.values) in emotionCount:
                if emotionCount[str(emotion.values)] > -1:
                    feeling_list.append(emotion.values)
                    utterances.append(utt.values)
                    emotionCount[str(emotion.values)] += 1
            else:
                emotionCount[str(emotion.values)] = 1
                feeling_list.append(emotion.values)
                utterances.append(utt.values)
            print(emotionCount)

        if currentLength < len(feeling_list):
            labelledFiles.append(itemPath)

    # Store the labels in a DataFrame made from the above feeling list
    labels = pd.DataFrame(feeling_list)
    utter = pd.DataFrame(utterances, columns=["feature"])
    df1 = createFeatureLabelDataFrame(utter, labels)
    df1.to_csv("../Datasets/MELD/utterancesFull.csv", index=False)
    # Print the length of both our labels and our usable files
    print(len(feeling_list), "Labels")
    print(len(labelledFiles), "Usable Files")

    return labels, labelledFiles


# Extract the relevant features from the audio samples
def extractFeatures(fileList, res_type, duration, sampleRate, startOffset, inMono, num_mfcc):
    df = pd.DataFrame(columns=['feature'])
    bookmark = 0
    for index, y in enumerate(fileList):
        X, sample_rate = librosa.load(y, res_type=res_type, duration=duration, sr=sampleRate, offset=startOffset, mono=inMono)
        sample_rate = np.array(sample_rate)
        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
        zcR = np.mean(librosa.feature.zero_crossing_rate(y=X).T, axis=0)
        chroma = np.mean(librosa.feature.chroma_cens(y=X).T, axis=0)
        feature = np.concatenate((mfccs, zcR))
        feature = np.concatenate((feature, chroma))
        #[float(i) for i in feature]
        #feature1=feature[:135]
        df.loc[bookmark] = [feature]
        bookmark = bookmark+1

    # Display 5 entries of the features
    print("Feature Extraction:")
    print(df[:5])
    return df


# BASELINE FEATURES Extract the relevant features from the audio samples
def baselineExtractFeatures(fileList, res_type, duration, sampleRate, startOffset, inMono, num_mfcc):
    df = pd.DataFrame(columns=['feature'])
    bookmark = 0
    for index, y in enumerate(fileList):
        X, sample_rate = librosa.load(y, res_type=res_type, duration=duration, sr=sampleRate, offset=startOffset, mono=inMono)
        sample_rate = np.array(sample_rate)
        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
        df.loc[bookmark] = [mfccs]
        bookmark = bookmark+1

    # Display 5 entries of the features
    print("Feature Extraction:")
    print(df[:5])
    return df


# Combine  features and labels DataFrames into one
def createFeatureLabelDataFrame(features, labels):
    # Adds features and labels to one DataFrame
    df = pd.DataFrame(features['feature'].values.tolist())
    labels = labels.rename(columns={0: "label"})
    combinedDf = pd.concat([df, labels], axis=1)
    # Shuffle the DataFrame
    #combinedDf = shuffle(combinedDf)
    # Fill any NA values with 0
    combinedDf = combinedDf.fillna(0)

    print(combinedDf)
    return combinedDf


# Return the train/test split and fitted encoder
def createTrainTestFromDataset(dataset, trainTestSplit):
    # Split features and labels
    datasetCopy = dataset
    labels = datasetCopy['label']
    features = datasetCopy.drop(columns='label')

    # Split the train/test by passed percentage
    # Split into train and test
    trainIndex = int(len(features) * trainTestSplit)
    train_features = features[:trainIndex]
    train_labels = labels[:trainIndex]
    test_features = features[trainIndex + 1:-1]
    test_labels = labels[trainIndex + 1:-1]

    lb = LabelEncoder()
    y_train = np_utils.to_categorical(lb.fit_transform(train_labels))
    y_test = np_utils.to_categorical(lb.fit_transform(test_labels))

    return train_features, test_features, y_train, y_test, lb


# Create the affective prosody model (returns model)
def createAPModel(inputDim, outputDim, lossMethod, optimizer, metricsList):
    model = Sequential()
    model.add(Conv1D(256, 5, padding='same',
                     input_shape=inputDim))
    model.add(Activation('relu'))
    model.add(Conv1D(128, 5, padding='same'))
    model.add(Activation('relu'))
    model.add(Dropout(0.1))
    model.add(MaxPooling1D(pool_size=(8)))
    model.add(Conv1D(64, 5, padding='same'))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(outputDim))
    model.add(Activation('softmax'))
    model.compile(loss=lossMethod, optimizer=optimizer, metrics=metricsList)
    return model


# Create the new Baseline affective prosody model (returns model)
def createNewBaseAPModel(inputDim, outputDim, lossMethod, optimizer, metricsList):
    model = Sequential()
    model.add(Conv1D(128, 5, padding='same',
                     input_shape=inputDim))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(MaxPooling1D(pool_size=(8)))
    model.add(Conv1D(128, 5, padding='same'))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(outputDim))
    model.add(Activation('softmax'))
    print(model.summary())
    model.compile(loss=lossMethod, optimizer=optimizer, metrics=metricsList)
    return model


# Create the affective prosody model (returns model)
def createBASELINEAPModel(inputDim, outputDim, lossMethod, optimizer, metricsList):
    model = Sequential()

    model.add(Conv1D(256, 5, padding='same',
                     input_shape=inputDim))
    model.add(Activation('relu'))
    model.add(Conv1D(128, 5, padding='same'))
    model.add(Activation('relu'))
    model.add(Dropout(0.1))
    model.add(MaxPooling1D(pool_size=(8)))
    model.add(Conv1D(128, 5, padding='same', ))
    model.add(Activation('relu'))
    model.add(Conv1D(128, 5, padding='same', ))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(outputDim))
    model.add(Activation('softmax'))
    model.compile(loss=lossMethod, optimizer=optimizer, metrics=metricsList)
    return model

# Fit the model to our data (returns the model history)
def model_fit_CNN(model, x_train, y_train, batchSize, epochs, x_test, y_test):
    # Changing Dimension for CNN model
    x_train = np.expand_dims(x_train, axis=2)
    x_test = np.expand_dims(x_test, axis=2)
    cnnhistory = model.fit(x_train, y_train, batch_size=batchSize, epochs=epochs, validation_data=(x_test, y_test))
    return cnnhistory


# Predict classifications returns a DataFrame of predicted labels (Strings)
def model_predict(model, x_test, batchSize, verbosity, labelEncoder):
    # Changing Dimension for CNN model
    x_test = np.expand_dims(x_test, axis=2)

    # Get predictions
    preds = model.predict(x_test, batch_size=batchSize, verbose=verbosity)
    predWeights = preds
    preds = preds.argmax(axis=1)
    og_preds = preds.astype(int).flatten()
    preds = (labelEncoder.inverse_transform(og_preds))
    return pd.DataFrame({'predictedValues': preds}), og_preds, predWeights


# Feature extraction and file labelling procedure RAVDESS
def create_RAVDESS_dataset(savePath, numC, isBaseline):
    if savePath.find("RAVDESS") != -1:
        wavFilePaths = getAllWavFiles("../Datasets/RAVDESS")
        labelList, labelledFilePaths = labelRAVDESS(wavFilePaths)
    else:
        wavFilePaths = getAllWavFiles("../Datasets/SAVEE/archive/AudioData/AudioData")
        labelList, labelledFilePaths = label_SAVEE(wavFilePaths)
    if isBaseline:
        featuresList = baselineExtractFeatures(labelledFilePaths, "kaiser_best", 3, 48000, 0, True, numC)
    else:
        featuresList = extractFeatures(labelledFilePaths, "kaiser_best", 3, 48000, 0, True, numC)
    labelledFeatures = createFeatureLabelDataFrame(featuresList, labelList)
    if savePath:
        pd.DataFrame(labelledFeatures).to_csv(savePath, index=False)
    return labelledFeatures


# Feature extraction and file labelling procedure EMO_DB
def create_EMODB_dataset(savePath, numC, isBaseline):
    wavFilePaths = getAllWavFiles("../Datasets/EMO_DB")
    labelList, labelledFilePaths = label_EMO_DB(wavFilePaths)
    if isBaseline:
        featuresList = baselineExtractFeatures(labelledFilePaths, "kaiser_best", 8, 48000, 0, True, numC)
    else:
        featuresList = extractFeatures(labelledFilePaths, "kaiser_best", 8, 48000, 0, True, numC)
    labelledFeatures = createFeatureLabelDataFrame(featuresList, labelList)
    if savePath:
        pd.DataFrame(labelledFeatures).to_csv(savePath, index=False)
    return labelledFeatures


# Feature extraction and file labelling procedure MELD
def create_MELD_dataset(savePath, numC, isBaseline):
    mp3FilePaths = getAllMP3Files("E:\FYP-Implementation\Datasets\MELD")
    '''for mp4 in mp4FilePaths:
        clip = mp.VideoFileClip(mp4)
        clip.audio.write_audiofile("../Datasets/MELD/"+str(os.path.basename(mp4).replace(".mp4", ".mp3")))'''
    labelList, labelledFilePaths = label_MELD(mp3FilePaths)
    if isBaseline:
        featuresList = baselineExtractFeatures(labelledFilePaths, "kaiser_best", 3, 48000, 0, True, numC)
    else:
        featuresList = extractFeatures(labelledFilePaths, "kaiser_best", 3, 48000, 0, True, numC)
    labelledFeatures = createFeatureLabelDataFrame(featuresList, labelList)
    if savePath:
        pd.DataFrame(labelledFeatures).to_csv(savePath, index=False)
    return labelledFeatures


# Encode the classification labels, return both the labels and the encoder
def encodeLabels(dataset):
    lb = LabelEncoder()
    labels = np_utils.to_categorical(lb.fit_transform(dataset['label']))
    return labels, lb


# Train, Test the model, plot the accuracy results
def runKFoldAPModel(plotName, featureData, labelData, encoder, oldLabel, isBaseline, mHistDF):
    # K-fold init
    kf = model_selection.KFold(n_splits=5, shuffle=True)
    iteration = 0
    totalAcc = 0
    bestAcc = 0
    bestModel = ""
    mX = []
    mY = []
    os.mkdir("Model Results/"+plotName)
    f = open("Model Results/"+plotName+"/modelRes.txt", "w")

    for trainIndex, testIndex, in kf.split(featureData):

        xTrain = featureData[trainIndex]
        yTrain = labelData[trainIndex]
        xTest = featureData[testIndex]
        yTest = labelData[testIndex]
        print(xTrain.shape)
        print(xTest.shape)
        print(yTrain.shape)
        print(yTest.shape)
        if isBaseline:
            AP_Model = createNewBaseAPModel((xTrain.shape[1], 1), yTrain.shape[1], "categorical_crossentropy", "rmsprop", ['accuracy'])
        else:
            #AP_Model = createAPModel((xTrain.shape[1], 1), yTrain.shape[1], "categorical_crossentropy", "adam", ['accuracy'])
            AP_Model = createNewBaseAPModel((xTrain.shape[1], 1), yTrain.shape[1], "categorical_crossentropy", "rmsprop", ['accuracy'])
        modelHistory = model_fit_CNN(AP_Model, xTrain, yTrain, 16, 1000, xTest, yTest)
        t = time.time()
        t = time.ctime(t)
        details = {}
        details["date"] = t
        details["model"] = plotName+str(iteration)
        details["loss"] = [modelHistory.history["loss"]]
        details["accuracy"] = [modelHistory.history["accuracy"]]
        details["val_loss"] = [modelHistory.history["val_loss"]]
        details["val_accuracy"] = [modelHistory.history["val_accuracy"]]
        details["train_length"] = len(xTrain)
        details["test_length"] = len(xTest)

        mHistDF = mHistDF.append(pd.DataFrame.from_dict(details))


        plot_history(plotName+str(iteration), modelHistory, "Model Results/"+plotName+"/")

        predictions, actualPreds, predictionWeights = model_predict(AP_Model, xTest, 16, 1, encoder)
        mX.extend(predictionWeights)
        mY.extend(oldLabel[testIndex])
        print("\nPredictions:\n", predictions)
        score = metrics.accuracy_score(labelData[testIndex].argmax(axis=1), actualPreds)
        if score > bestAcc:
            bestModel = AP_Model
            bestAcc = score
        totalAcc += score
        print("Model Accuracy:", score)
        print("Conf Matrix:\n", metrics.confusion_matrix(oldLabel[testIndex], predictions))
        print(metrics.classification_report(oldLabel[testIndex], predictions))
        f.writelines(np.array2string(metrics.confusion_matrix(oldLabel[testIndex], predictions)))
        f.write("\n")
        f.writelines(metrics.classification_report(oldLabel[testIndex], predictions))
        f.write("\nAccuracy: "+str(score)+"\n")
        clear_session()
        # Calculate the spectrum of emotions
        for val in predictionWeights:
            total = sum(val)
            percentages = []
            for v in val:
                percentages.append(str(round((v/total) * 100, 2)) + "%")
            #print(percentages)
        iteration += 1
    print("Avg Accuracy:", totalAcc/5)
    f.write("Avg Accuracy: " + str(totalAcc/5) + "\n")
    f.close()
    return bestModel, mHistDF, mX, np.array(mY)


# Train, Test the combined model system
def runKFoldCombinedModel(plotName, featureData, labelData, encoder, oldLabel, isBaseline, mHistDF, logRegModel, textData, vectorizerObj, textLabelEncoder):
    # K-fold init
    kf = model_selection.KFold(n_splits=5, shuffle=True)
    iteration = 0
    totalAcc = 0
    bestAcc = 0
    bestModel = ""
    mX = []
    mY = []
    os.mkdir("Model Results/"+plotName)
    f = open("Model Results/"+plotName+"/modelRes.txt", "w")

    for trainIndex, testIndex, in kf.split(featureData):

        xTrain = featureData[trainIndex]
        yTrain = labelData[trainIndex]
        xTest = featureData[testIndex]
        yTest = labelData[testIndex]
        textXTest = textData[testIndex]
        if isBaseline:
            AP_Model = createBASELINEAPModel((xTrain.shape[1], 1), yTrain.shape[1], "categorical_crossentropy", "adam", ['accuracy'])
        else:
            AP_Model = createAPModel((xTrain.shape[1], 1), yTrain.shape[1], "categorical_crossentropy", "adam", ['accuracy'])
        modelHistory = model_fit_CNN(AP_Model, xTrain, yTrain, 16, 50, xTest, yTest)
        t = time.time()
        t = time.ctime(t)
        details = {}
        details["date"] = t
        details["model"] = plotName+str(iteration)
        details["loss"] = [modelHistory.history["loss"]]
        details["accuracy"] = [modelHistory.history["accuracy"]]
        details["val_loss"] = [modelHistory.history["val_loss"]]
        details["val_accuracy"] = [modelHistory.history["val_accuracy"]]
        details["train_length"] = len(xTrain)
        details["test_length"] = len(xTest)

        mHistDF = mHistDF.append(pd.DataFrame.from_dict(details))
        plot_history(plotName+str(iteration), modelHistory, "Model Results/"+plotName+"/")

        predictions, actualPreds, predictionWeights = model_predict(AP_Model, xTest, 16, 1, encoder)

        textXTest = vectorizerObj.transform(textXTest)
        textPreds = logRegModel.predict(textXTest)
        textPreds = np_utils.to_categorical(lb.fit_transform(textPreds))
        mergedPreds = np.concatenate((predictionWeights, textPreds), axis=1)
        mX.extend(mergedPreds)
        mY.extend(oldLabel[testIndex])
        print("\nPredictions:\n", predictions)
        score = metrics.accuracy_score(labelData[testIndex].argmax(axis=1), actualPreds)
        if score > bestAcc:
            bestModel = AP_Model
            bestAcc = score
        totalAcc += score
        print("Model Accuracy:", score)
        print("Conf Matrix:\n", metrics.confusion_matrix(oldLabel[testIndex], predictions))
        print(metrics.classification_report(oldLabel[testIndex], predictions))
        f.writelines(np.array2string(metrics.confusion_matrix(oldLabel[testIndex], predictions)))
        f.write("\n")
        f.writelines(metrics.classification_report(oldLabel[testIndex], predictions))
        f.write("\nAccuracy: "+str(score)+"\n")
        clear_session()
        # Calculate the spectrum of emotions
        for val in predictionWeights:
            total = sum(val)
            percentages = []
            for v in val:
                percentages.append(str(round((v/total) * 100, 2)) + "%")
            #print(percentages)
        iteration += 1
    print("Avg Accuracy:", totalAcc/5)
    f.write("Avg Accuracy: " + str(totalAcc/5) + "\n")
    f.close()
    return bestModel, mHistDF, mX, np.array(mY)
'''
*------------------------------------------------*
|                                                |
|   NLP / Text Model Functions                   |
|                                                |
*------------------------------------------------*
'''


# Extract the sentences and their labels from a dataset
def extractTextFeaturesAndLabels(dataset):
    sentences = dataset["text"].values
    labels = dataset['label'].values
    return sentences, labels


# Change labels into categorical values (returns train labels, test labels and the encoder)
def encodeTextLabels(trainLabels, testLabels):
    lb = LabelEncoder()
    trainLabels = np_utils.to_categorical(lb.fit_transform(trainLabels))
    testLabels = np_utils.to_categorical(lb.fit_transform(testLabels))
    return trainLabels, testLabels, lb


# Create the vectorizer object, fitted to the input data
def createVectorizer(lowercase, dataToFit):
    # Vectorizer Definition
    vectorizer = CountVectorizer(min_df=0, lowercase=lowercase)
    vectorizer.fit(dataToFit)
    return vectorizer


# Convert the train and test sentences into vectors
def vectorizeSentences(trainSentences, testSentences, vectorizer):
    X_train = vectorizer.transform(trainSentences)
    X_test = vectorizer.transform(testSentences)
    return X_train, X_test


# Create and return a fitted LogisticRegression classifier
def createLogRegTextClassifier(maxIterations, X_train, Y_train):
    classifier = LogisticRegression(max_iter=maxIterations)
    classifier.fit(X_train, Y_train)
    return classifier


# Predict the classes using the classifier outputs: amount of predictions, accuracy score and confusion matrix
def predictText(classifier, X_test, Y_test):
    preds = classifier.predict(X_test)
    print(len(preds))
    print("LogReg Accuracy:", accuracy_score(Y_test, preds))
    print("Confusion Matrix:")
    print(multilabel_confusion_matrix(Y_test, preds))


# Train, Test the model, plot the accuracy results
def runKFoldTextModel(featureData, labelData, isCNN):
    # K-fold init
    kf = model_selection.KFold(n_splits=5, shuffle=True)
    iteration = 0
    for trainIndex, testIndex, in kf.split(featureData):

        xTrain = featureData[trainIndex]
        yTrain = labelData[trainIndex]
        xTest = featureData[testIndex]
        yTest = labelData[testIndex]

        # yTrain, yTest, encoder = encodeTextLabels(yTrain, yTest)
        vectorizerObj = createVectorizer(False, xTrain)
        xTrain, xTest = vectorizeSentences(xTrain, xTest, vectorizerObj)

        model = createLogRegTextClassifier(1000, xTrain, yTrain)
        predictText(model, xTest, yTest)

        clear_session()
        iteration += 1

    return model, vectorizerObj


def testTextClassifier(dataset, classifier, vectorizer):
    if dataset == "../Datasets/RAVDESS":
        # For further testing get all ravdess files
        files = getAllWavFiles("../Datasets/RAVDESS")
        labels, labelledFiles = labelRAVDESS(files)
        werTotal = 0
        c = 0
        w = 0
        for f in labelledFiles:
            res = transcribe(f)
            pred = classifier.predict(vectorizer.transform([res]))
            emo, txt = returnFileInfoRAVDESS(f)
            werTotal += calculateWER(res, txt)
            cleanEmo = emo[0][0].split("_")
            if cleanEmo[1] == pred[0]:
                c += 1
            else:
                w += 1
            print("\nTranscription")
            print("Result:", res)
            print("Actual:", txt)
            print("\nEmotion Classification")
            print("Result:", pred[0])
            print("Actual:", emo[0][0])

        print("\nWER TOTAL:", werTotal)
        print(len(labelledFiles) / werTotal)

    elif dataset == "../Datasets/SAVEE":
        # For further testing get all ravdess files
        files = getAllWavFiles("../Datasets/SAVEE")
        labels, labelledFiles = label_SAVEE(files)
        werTotal = 0
        c = 0
        w = 0
        for f in labelledFiles:
            res = transcribe(f)
            pred = classifier.predict(vectorizer.transform([res]))
            emo = returnFileInfoSAVEE(f)
            #werTotal += calculateWER(res, txt)
            if emo[0][0] == pred[0]:
                c += 1
            else:
                w += 1
            print("\nTranscription")
            print("Result:", res)
            #print("Actual:", txt)
            print("\nEmotion Classification")
            print("Result:", pred[0])
            print("Actual:", emo[0][0])


    else:
        # For further testing get all ravdess files
        files = getAllWavFiles("../Datasets/EMO_DB")
        labels, labelledFiles = label_EMO_DB(files)
        c = 0
        w = 0
        for f in labelledFiles:
            res = translate_EMODB_file(f)
            pred = classifier.predict(vectorizer.transform([res]))
            temp = labels[0][c+w].split("_")
            print("\nTranslation")
            print("Result:", res)
            print("\nEmotion Classification")
            print("Result:", pred[0])
            print("Actual:", temp)
            if pred[0] == temp[1]:
                c += 1
            else:
                w += 1

    print("Correct Emotions:", c)
    print("Wrong Emotions:", w)
    print("Accuracy:", c / (w + c))


'''
*------------------------------------------------*
|                                                |
|   Test Affective Prosody Models                |
|                                                |
*------------------------------------------------*
'''
# M1 - RAVDESS -> CNN -> Emotional Classification
#create_RAVDESS_dataset("../Datasets/RAVDESS/mfcc3.csv")
#df = pd.read_csv("../Datasets/RAVDESS/mfcc3.csv")
#ravdessLabels = ["female_fearful", "male_fearful", "female_angry", "male_angry", "female_sad", "male_sad", "female_happy", "male_happy"]

#create_RAVDESS_dataset("../Datasets/SAVEE/mfcc3.csv")
#df = pd.read_csv("../Datasets/SAVEE/bestFeatures.csv")

#create_MELD_dataset("../Datasets/MELD/test.csv")
#df = pd.read_csv("../Datasets/MELD/test.csv")

# # Create and fit the logisitc regression model on the ISEARS dataset
# textTrainData = pd.read_csv("../Datasets/ISEAR.csv") # Emotion Range: [joy, fear, anger, sadness, disgust, shame, guilt]
# # Remove \n chars
# textTrainData["text"] = textTrainData["text"].str.replace("\n", "")
# # Remove unsupported emotions
# testableData = textTrainData[textTrainData["label"] != "shame"]
# testableData = testableData[testableData["label"] != "guilt"]
# testableData["label"] = testableData["label"].replace("sadness", "sad")
#
# # Fitting Log Reg Model to ISEARS Data
# sentences, Textlabels = extractTextFeaturesAndLabels(testableData)
# vec = createVectorizer(False, sentences)
# trainingFeatures = vec.transform(sentences)
#
# lb = LabelEncoder()
# #trainLabels = np_utils.to_categorical(lb.fit_transform(Textlabels))
# logRegModel = createLogRegTextClassifier(1000, trainingFeatures, Textlabels)

datasetNames = ["../Datasets/RAVDESS/baselineNew.csv"]
for dataset in datasetNames:
    if dataset.find("baseline") != -1:
        baseline = True
        ext = "Baseline-"
        numC = 13
    else:
        baseline = False
        numC = 8
        ext = ""

    if dataset.find("RAVDESS") != -1:
        create_RAVDESS_dataset(dataset, numC, baseline)
        name = "RAVDESS-" + ext
    elif dataset.find("EMO_DB") != -1:
        #create_EMODB_dataset(dataset, numC, baseline)
        name = "EMO_DB-"+ ext
    elif dataset.find("SAVEE") != -1:
        create_RAVDESS_dataset(dataset, numC, baseline)
        name = "SAVEE-"+ ext
    else:
        #create_MELD_dataset(dataset, numC, baseline)
        name = "MELD-"+ ext

    # df = pd.read_csv(dataset)
    # oldLabel = df['label']
    # print("Class Distribution:")
    # print(df['label'].value_counts())
    # labelDataFrame, lEncoder = encodeLabels(df)
    # featureDataFrame = df.drop(columns='label').to_numpy()
    #
    # bestModel, hist, x, y = runKFoldAPModel(name, featureDataFrame, labelDataFrame, lEncoder, oldLabel, True, modelHistoryData)
    # modelHistoryData = hist
    # df = pd.read_csv(dataset)
    # textdfName ="../Datasets/"+name[:len(name)-1]+"/utterancesFull.csv"
    # textdf = pd.read_csv(textdfName)
    # textdf, unusedLabels = extractTextFeaturesAndLabels(textdf)
    #
    # print("Class Distribution:")
    # print(df['label'].value_counts())
    # oldLabel = df['label']
    # labelDataFrame, lEncoder = encodeLabels(df)
    # featureDataFrame = df.drop(columns='label').to_numpy()
    #
    # # META Train Test Split
    # splitIndex = int(len(featureDataFrame) * 0.85)
    # metaValX = featureDataFrame[splitIndex + 1:-1]
    # metaValText = textdf[splitIndex + 1:-1]
    # metaValY = oldLabel[splitIndex + 1:-1]
    #
    # best_Model, hist, metaTrainX, metaTrainY = runKFoldCombinedModel(name, featureDataFrame[:splitIndex], labelDataFrame[:splitIndex], lEncoder, oldLabel, True, modelHistoryData, logRegModel, textdf[:splitIndex], vec, lb)
    # modelHistoryData = hist
    #
    # metaModel = LogisticRegression()
    # metaModel.fit(metaTrainX, metaTrainY)
    #
    # preds, acPreds, truePredictions = model_predict(best_Model, metaValX, 16, 1, lEncoder)
    # metaValText = vec.transform(metaValText)
    # textPreds = logRegModel.predict(metaValText)
    # textPreds = np_utils.to_categorical(lb.fit_transform(textPreds))
    # mergedPreds = np.concatenate((truePredictions, textPreds), axis=1)
    #
    # meta_preds = metaModel.predict(mergedPreds)
    # score = metrics.accuracy_score(metaValY, meta_preds)
    # print("Meta Model Score:", score)
#modelHistoryData.to_csv("Model Results/modelDetails.csv", index=False)




#df1 = pd.read_csv("../Datasets/SAVEE/bestFeatures.csv")
#print("Class Distribution:")
#print(df1['label'].value_counts())
#oldLabel = df1['label']
#featureDataFrame = df1.drop(columns='label').to_numpy()
#labelDataFrame = np_utils.to_categorical(lEncoder.fit_transform(oldLabel))
#preds, accPreds, weights = model_predict(best_Model, featureDataFrame, 32, 1, lEncoder)
#print("\nPredictions:\n", preds)
#score = metrics.accuracy_score(labelDataFrame.argmax(axis=1), accPreds)
#print("Model Accuracy:", score)


# M2 - EMO DB -> CNN -> Emotional Classification

#create_EMODB_dataset("../Datasets/EMO_DB/mfcc-allClasses.csv")
# emoDBLabels = ["female_anger", "male_anger", "female_boredom", "male_boredom", "female_happy", "male_happy",
#                 "female_neutral", "male_neutral", "female_sad", "male_sad", "female_disgust", "male_disgust",
#                 "female_anxiety/fear", "male_anxiety/fear"]

#df = pd.read_csv("../Datasets/EMO_DB/mfcc-allClasses.csv")

#print("Class Distribution:")
#print(df['label'].value_counts())
#oldLabel = df['label']
#labelDataFrame, lEncoder = encodeLabels(df)
#featureDataFrame = df.drop(columns='label').to_numpy()
#runKFoldAPModel("EMO_DB-", featureDataFrame, labelDataFrame, lEncoder, oldLabel)


'''
*------------------------------------------------*
|                                                |
|   Test Text Classification Models              |
|                                                |
*------------------------------------------------*
'''

# Read in the emotional text dataset
#data = pd.read_csv("../Datasets/ISEAR.csv") # Emotion Range: [joy, fear, anger, sadness, disgust, shame, guilt]
# Remove \n chars
#data["text"] = data["text"].str.replace("\n", "")
#print(data["label"].unique())
# Remove unsupported emotions
#testableData = data[data["label"] != "shame"]
#testableData = testableData[testableData["label"] != "guilt"]
#testableData["label"] = testableData["label"].replace("sadness", "sad")

# Convert the labels to numbers
# print(testableData["label"].unique())
# testableData["label"] = testableData["label"].replace("joy", 1)
# testableData["label"] = testableData["label"].replace("fear", 2)
# testableData["label"] = testableData["label"].replace("anger", 3)
# testableData["label"] = testableData["label"].replace("sadness", 4)
# testableData["label"] = testableData["label"].replace("disgust", 5)

# Extract features and labels
#features, labels = extractTextFeaturesAndLabels(testableData)

# Create and train the model
#m, vec = runKFoldTextModel(features, labels, False)


#testTextClassifier("../Datasets/SAVEE", m, vec)

