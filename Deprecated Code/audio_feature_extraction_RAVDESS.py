import os
import pandas as pd
import librosa
import numpy as np
from sklearn.utils import shuffle


# Get all audio files in dirs and sub dirs
def getAllWavFiles(pathToSearch):
    filenameList = []
    filePathList = []
    for root, dirs, files in os.walk(pathToSearch):
        for file in files:
            # append the file name to the list
            if file.endswith(".wav"):
                filePathList.append(os.path.join(root, file))
                filenameList.append(file)

    print("Found", len(filenameList), "WAV files")
    return filenameList, filePathList


# Setting Labels based on filenames
def labelFilesBasedOnFilename(filenameList, filePathList):
    feeling_list=[]
    labelledFiles = []
    for item in filenameList:
        currentLength = len(feeling_list)
        if item[6:-16]=='02' and int(item[18:-4])%2==0:
            feeling_list.append('calm')
        elif item[6:-16]=='02' and int(item[18:-4])%2==1:
            feeling_list.append('calm')
        elif item[6:-16]=='03' and int(item[18:-4])%2==0:
            feeling_list.append('joy')
        elif item[6:-16]=='03' and int(item[18:-4])%2==1:
            feeling_list.append('joy')
        elif item[6:-16]=='04' and int(item[18:-4])%2==0:
            feeling_list.append('sadness')
        elif item[6:-16]=='04' and int(item[18:-4])%2==1:
            feeling_list.append('sadness')
        elif item[6:-16]=='05' and int(item[18:-4])%2==0:
            feeling_list.append('anger')
        elif item[6:-16]=='05' and int(item[18:-4])%2==1:
            feeling_list.append('anger')
        elif item[6:-16]=='06' and int(item[18:-4])%2==0:
            feeling_list.append('fear')
        elif item[6:-16]=='06' and int(item[18:-4])%2==1:
            feeling_list.append('fear')
        elif item[:1]=='a':
            feeling_list.append('anger')
        elif item[:1]=='f':
            feeling_list.append('fear')
        elif item[:1]=='h':
            feeling_list.append('joy')
        #elif item[:1]=='n':
            #feeling_list.append('neutral')
        elif item[:2]=='sa':
            feeling_list.append('sadness')
        if currentLength < len(feeling_list):
            labelledFiles.append(filePathList[filenameList.index(item)])

    # Store the labels in a DataFrame made from the above feeling list
    labels = pd.DataFrame(feeling_list)

    # Print the length of both our labels and our usable files
    print(len(feeling_list), "Labels")
    print(len(labelledFiles), "Usable Files")

    return labels, labelledFiles


# Extract the relevant features from the audio samples
def extractFeatures(fileList):
    df = pd.DataFrame(columns=['feature'])
    bookmark = 0
    for index,y in enumerate(fileList):
        if fileList[index][6:-16]!='01' and fileList[index][6:-16]!='07' and fileList[index][6:-16]!='08' and fileList[index][:2]!='su' and fileList[index][:1]!='n' and fileList[index][:1]!='d':
            X, sample_rate = librosa.load(y, res_type='kaiser_fast',duration=2.5,sr=22050*2,offset=0.5, mono=True)
            sample_rate = np.array(sample_rate)
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13), axis=0)
            feature = mfccs
            #[float(i) for i in feature]
            #feature1=feature[:135]
            df.loc[bookmark] = [feature]
            bookmark = bookmark+1

    # Display 5 entries of the features
    print("Feature Extraction:")
    print(df[:5])

    return df


def createFeatureLabelDataFrame(features, labels):
    # Adds features and labels to one DataFrame
    df = pd.DataFrame(features['feature'].values.tolist())
    labels = labels.rename(columns={0: "label"})
    combinedDf = pd.concat([df, labels], axis=1)
    # Shuffle the dataframe
    combinedDf = shuffle(combinedDf)
    # Fill any NA values with 0
    combinedDf = combinedDf.fillna(0)
    print(combinedDf)

    return combinedDf


def getFeaturesAndLabels(path, isSaved):

    # Get all WAV filenames and paths
    filenames, filePaths = getAllWavFiles(path)

    # Create a DataFrame of labels based on filenames
    labels, usableFiles = labelFilesBasedOnFilename(filenames, filePaths)

    # Extract the features from all of the files that have labels (usableFiles)
    features = extractFeatures(usableFiles)

    # Create a single DataFrame including the features and the labels
    dataset = createFeatureLabelDataFrame(features, labels)

    if isSaved:
        pd.DataFrame(dataset).to_csv("../Datasets/RAVDESS\\speechActorDataset2.csv", index=False)

    return dataset


# Define the path where our WAV files are stored
ourPath = "..\\Datasets\\Speech Emotion Analyzer\\RAVDESS\\Audio_Speech_Actors_01-24"
getFeaturesAndLabels(ourPath, True)
