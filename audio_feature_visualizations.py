import matplotlib.pyplot as plt
import librosa
import librosa.display

# Load the audio file
filepath = "../Datasets/RAVDESS/Audio_Speech_Actors_01-24/Actor_01/03-01-01-01-01-01-01.wav"
vector, sr = librosa.load(filepath, duration=3, res_type="kaiser_best", sr=48000)

# Visualize the audio signal
plt.plot(vector)
plt.xlabel("Sample Rate")
plt.ylabel("Amplitude")
plt.title("Audio Signal")
plt.show()

# Visualize the MFFCs
mfccs = librosa.feature.mfcc(y=vector, sr=sr, n_mfcc=40)
fig, ax = plt.subplots()
img = librosa.display.specshow(mfccs, x_axis='time', ax=ax)
fig.colorbar(img, ax=ax)
ax.set(title='Mel-frequency cepstral coefficients')
plt.show()

# Visualize the ZCR
ZCR = librosa.feature.zero_crossing_rate(y=vector)
fig, ax = plt.subplots()
img = librosa.display.specshow(ZCR, x_axis='time', ax=ax)
fig.colorbar(img, ax=ax)
ax.set(title='Zero Crossing Rate')
plt.show()

# Visualize the CC
CC = librosa.feature.chroma_cens(y=vector)
fig, ax = plt.subplots()
img = librosa.display.specshow(CC, x_axis='time', ax=ax)
fig.colorbar(img, ax=ax)
ax.set(title='Chroma Cens')
plt.show()