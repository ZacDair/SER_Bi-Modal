"""
This file outlines the data flow of the workflow

1) Retrieve audio input, either live or as a file
*2) Split processing power into the two modalities
3) Extract the text content using transcription
4.1) Emotive analysis of the audio
4.2) Emotive analysis of the text
5) Sentiment analysis of the text
6.1) Semantic analysis of the text
6.2) Semantic analysis of the audio
... Additional analysis stages here....
7) Contextual ???
8) Merge and display the end analysis
"""
import librosa
import numpy as np
import speech_recognition as sr
import model_io_functions as mIO

#
#inputAudio = getInput() # from user mic or file
#textContent = transcribe(inputAudio) # load up transcription method
#emotion_AP = classifyAffectiveProsody(inputAudio) # load up AP Model
#emotion_NLP = classifyText(textContent) # load up Text Model


# Extract Affective Prosody Features
def extractFeatures_AP(filepath):
    X, sample_rate = librosa.load(filepath, res_type='kaiser_fast', duration=2.5, sr=22050*2, offset=0.5, mono=True)
    sample_rate = np.array(sample_rate)
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13), axis=0)
    x_traincnn = np.expand_dims([mfccs], axis=2)
    return x_traincnn


# Transcribe a WAV file to text and return the result
def transcribe(audioFilePath):
    audioFile = sr.AudioFile(audioFilePath)

    # Define the speech recognising method (Google, IBM..etc)
    r = sr.Recognizer()

    # Use the audio file as the source
    with audioFile as source:
        audio = r.record(source)  # read the entire audio file

    return r.recognize_google(audio)


# Extract NLP features
def extractFeatures_NLP(filepath):
    textContent = transcribe(filepath)
    vectorizer = mIO.loadPickledModel("my_models/baseline_text_vectorizer.sav")
    features = vectorizer.transform([textContent])
    return features


# Get Affective Prosody Classification
def classify_AP(features):
    ap_model = mIO.loadJsonModel("my_models/affectiveProsody_model.h5", "my_models/affectiveProsody_model.json")
    encoder = mIO.loadPickledModel("my_models/baseline_AP_labelEncoder.sav")
    preds = ap_model.predict(features)
    preds = preds.argmax(axis=1)
    preds = preds.astype(int).flatten()
    predictions = (encoder.inverse_transform(preds))
    return predictions


# Get NLP Emotion Classification
def classify_NLP(features):
    nlp_model = mIO.loadPickledModel("my_models/baseline_text_emotion_classifier.sav")
    return nlp_model.predict(features)


path = "../Datasets/RAVDESS/Audio_Speech_Actors_01-24/Actor_01/03-01-06-01-01-01-01.wav"
emotion_AP = classify_AP(extractFeatures_AP(path))
emotion_NLP = classify_NLP(extractFeatures_NLP(path))

print("Resulting Emotional Classifications:")
print("Affective Prosody:", emotion_AP)
print("Text Content:", emotion_NLP)