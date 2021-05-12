import speech_recognition as sr
from os import walk, path
from jiwer import wer
from pydub import AudioSegment
import time


# Get all audio files in dirs and sub dirs
def getAllWavFiles(pathToSearch):
    filenameList = []
    filePathList = []
    for root, dirs, files in walk(pathToSearch):
        for file in files:
            # append the file name to the list
            if file.endswith(".wav"):
                filePathList.append(path.join(root, file))
                filenameList.append(file)

    print("Found", len(filenameList), "WAV files")
    return filenameList, filePathList


# Transcribe a WAV file to text and return the result
def transcribe(audioFilePath):
    audioFile = sr.AudioFile(audioFilePath)

    # Define the speech recognising method (Google, IBM..etc)
    r = sr.Recognizer()

    # Use the audio file as the source
    with audioFile as source:
        audio = r.record(source)  # read the entire audio file

    return r.recognize_google(audio)


# Utilise the naming convention to discern the spoken content (or labels)
def labelOurData(audioFilenames):
    statements = []
    for name in audioFilenames:
        nameSegments = name.split("-")
        if nameSegments[4] == "01":
            statement = "kids are talking by the door"
        elif nameSegments[4] == "02":
            statement = "dogs are sitting by the door"
        else:
            statement = "UNSURE"

        statements.append(statement)
    return statements


# Count the number of correct/wrong transcriptions (correct: res == actual)
def naiveAccuracy(transcribedLabels, actualLabels):
    i = 0
    correctCount = 0
    wrongCount = 0
    wrongList = []
    while i < len(actualLabels):
        if actualLabels[i] == transcribedLabels[i]:
            correctCount += 1
        else:
            wrongList.append([actualLabels[i], transcribedLabels[i]])
            wrongCount += 1
        i += 1
    return correctCount, wrongCount, wrongList


# Calculate the word error rate
def calculateWER(transcribedLabels, actualLabels):
    i = 0
    totalWer = 0
    while i < len(actualLabels):
        error = wer(actualLabels[i], transcribedLabels[i])
        totalWer += error
        print("error value:", error)
        i += 1
    print("Total:", totalWer)
    print("Divded by total count:", totalWer/len(actualLabels))


# Get all WAV filenames and paths
filenames, filePaths = getAllWavFiles("..\\Datasets\\Speech Emotion Analyzer\\RAVDESS\\Audio_Speech_Actors_01-24")

# Extract a portion of the total dataset
filenames = filenames[50:75]
filePaths = filePaths[50:75]

# Extract the label from the filename (label = lexical statement)
labels = labelOurData(filenames)


# Iterate through our files storing the result of the transcription
results = []
start = time.time()
for file in filePaths:
    results.append(transcribe(file))
print("Transcription of", len(results), "RAVDESS files took", time.time()-start)

# Output the number of correct/wrong transcriptions, and show the actual failed transcriptions
correct, wrong, wrongTranscriptions = naiveAccuracy(results, labels)
print("We got", correct, "correct and", wrong, "wrong transcriptions")
for x in wrongTranscriptions:
    print(x)

calculateWER(results, labels)