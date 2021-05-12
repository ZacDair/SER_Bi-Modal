from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten, Dropout, Activation
from keras.layers import Conv1D, MaxPooling1D
from sklearn import linear_model
from tensorflow.keras import optimizers
from keras.models import model_from_json
import librosa
import pandas as pd
import os
import numpy as np
from sklearn.utils import shuffle
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn import metrics

# load a trained model from a json file
def loadModel(weightFile, modelFile):
    try:
        jsonFile = open(modelFile, 'r')
        loadedModel = jsonFile.read()
        jsonFile.close()
        loadedModel = model_from_json(loadedModel)

        # load weights
        loadedModel.load_weights(weightFile)
        print("Loaded Model From Disk")
        return loadedModel
    except FileNotFoundError:
        print("ERROR - The imported model files were not found")
        exit(1)


# Speech-Emotion-Analyzer Model Files
weightFilePath = "../External Libs/Speech-Emotion-Analyzer/saved_models/Emotion_Voice_Detection_Model.h5"
modelFilePath = "../External Libs/Speech-Emotion-Analyzer/model.json"

# Try to load the model from the files
model = loadModel(weightFilePath, modelFilePath)

# Get all audio files
path = "../../Datasets/RAVDESS/Audio_Speech_Actors_01-24"
filenames = []
filePaths = []
for root, dirs, files in os.walk(path):
    for file in files:
        # append the file name to the list
        if file.endswith(".wav"):
            filePaths.append(os.path.join(root, file))
            filenames.append(file)

print("Found", len(filenames), "WAV files")

# Setting Labels based on filenames
feeling_list=[]
labelledFiles = []
for item in filenames:
    currentLength = len(feeling_list)
    if item[6:-16]=='02' and int(item[18:-4])%2==0:
        feeling_list.append('female_calm')
    elif item[6:-16]=='02' and int(item[18:-4])%2==1:
        feeling_list.append('male_calm')
    elif item[6:-16]=='03' and int(item[18:-4])%2==0:
        feeling_list.append('female_happy')
    elif item[6:-16]=='03' and int(item[18:-4])%2==1:
        feeling_list.append('male_happy')
    elif item[6:-16]=='04' and int(item[18:-4])%2==0:
        feeling_list.append('female_sad')
    elif item[6:-16]=='04' and int(item[18:-4])%2==1:
        feeling_list.append('male_sad')
    elif item[6:-16]=='05' and int(item[18:-4])%2==0:
        feeling_list.append('female_angry')
    elif item[6:-16]=='05' and int(item[18:-4])%2==1:
        feeling_list.append('male_angry')
    elif item[6:-16]=='06' and int(item[18:-4])%2==0:
        feeling_list.append('female_fearful')
    elif item[6:-16]=='06' and int(item[18:-4])%2==1:
        feeling_list.append('male_fearful')
    elif item[:1]=='a':
        feeling_list.append('male_angry')
    elif item[:1]=='f':
        feeling_list.append('male_fearful')
    elif item[:1]=='h':
        feeling_list.append('male_happy')
    #elif item[:1]=='n':
        #feeling_list.append('neutral')
    elif item[:2]=='sa':
        feeling_list.append('male_sad')
    if currentLength < len(feeling_list):
        labelledFiles.append(filePaths[filenames.index(item)])

# Store the labels in a DataFrame made from the above feeling list
labels = pd.DataFrame(feeling_list)

# Print the length of both our labels and our usable files
print(len(feeling_list), "Labels")
print(len(labelledFiles), "Usable Files")


# Shorten the file list for dev
#shortFileList = filePaths[:300]
shortFileList = labelledFiles

# Extract the features
df = pd.DataFrame(columns=['feature'])
bookmark = 0
for index,y in enumerate(shortFileList):
    if shortFileList[index][6:-16]!='01' and shortFileList[index][6:-16]!='07' and shortFileList[index][6:-16]!='08' and shortFileList[index][:2]!='su' and shortFileList[index][:1]!='n' and shortFileList[index][:1]!='d':
        X, sample_rate = librosa.load(y, mono=True, res_type='kaiser_fast',duration=2.5,sr=22050*2,offset=0.5)
        sample_rate = np.array(sample_rate)
        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=16), axis=0)
        feature = mfccs
        #[float(i) for i in feature]
        #feature1=feature[:135]
        df.loc[bookmark] = [feature]
        bookmark = bookmark+1

# Display 5 entries of the features
print("Feature Extraction:")
print(df[:5])

# Adds features and labels to one DataFrame
df3 = pd.DataFrame(df['feature'].values.tolist())
newdf = pd.concat([df3,labels], axis=1)
rnewdf = newdf.rename(index=str, columns={"0": "label"})

# Display 5 entries of the above
print("Features and Label Concatenation:")
print(rnewdf[:5])

# Shuffle the dataframe and display 10
rnewdf = shuffle(newdf)
print("Randomizing our data:")
print(rnewdf[:10])

# Fill any NA values with 0
rnewdf=rnewdf.fillna(0)


# Split into train and test
newdf1 = np.random.rand(len(rnewdf)) < 0.6
train = rnewdf[newdf1]
test = rnewdf[~newdf1]
trainfeatures = train.iloc[:, :-1]
trainlabel = train.iloc[:, -1:]
testfeatures = test.iloc[:, :-1]
testlabel = test.iloc[:, -1:]

X_train = np.array(trainfeatures)
y_train = np.array(trainlabel)
X_test = np.array(testfeatures)
y_test = np.array(testlabel)
stuff = y_test
lb = LabelEncoder()
y_train = np_utils.to_categorical(lb.fit_transform(y_train))
y_test = np_utils.to_categorical(lb.fit_transform(y_test))

print("Train/Test DataFrame Shapes:")
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)
# Define our perceptron model
perceptron = linear_model.Perceptron()

#perceptron.fit(X_train, y_train)
#predictions = perceptron.predict(X_test)
#score = metrics.accuracy_score(y_test, predictions)
#confusionMatrix = metrics.confusion_matrix(y_test, predictions)

# Changing Dimension for CNN model
x_traincnn =np.expand_dims(X_train, axis=2)
x_testcnn= np.expand_dims(X_test, axis=2)

model = Sequential()

model.add(Conv1D(256, 5,padding='same',
                 input_shape=(216,1)))
model.add(Activation('relu'))
model.add(Conv1D(128, 5,padding='same'))
model.add(Activation('relu'))
model.add(Dropout(0.1))
model.add(MaxPooling1D(pool_size=(8)))
model.add(Conv1D(128, 5,padding='same',))
model.add(Activation('relu'))
#model.add(Conv1D(128, 5,padding='same',))
#model.add(Activation('relu'))
#model.add(Conv1D(128, 5,padding='same',))
#model.add(Activation('relu'))
#model.add(Dropout(0.2))
model.add(Conv1D(128, 5,padding='same',))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(10))
model.add(Activation('softmax'))
opt = optimizers.RMSprop(lr=0.00001, decay=1e-6)

model.compile(loss='categorical_crossentropy', optimizer=opt,metrics=['accuracy'])
cnnhistory=model.fit(x_traincnn, y_train, batch_size=16, epochs=700, validation_data=(x_testcnn, y_test))

preds = model.predict(x_testcnn, batch_size=32, verbose=1)
preds1=preds.argmax(axis=1)
abc = preds1.astype(int).flatten()
predictions = (lb.inverse_transform((abc)))
preddf = pd.DataFrame({'predictedvalues': predictions})

plt.plot(cnnhistory.history['acc'])
plt.plot(cnnhistory.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


model_name = 'Emotion_Voice_Detection_Model.h5'
save_dir = os.path.join(os.getcwd(), 'saved_models')

# Save model and weights
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)
model_json = model.to_json()
with open(save_dir+"/model.json", "w") as json_file:
    json_file.write(model_json)

index = 0
correct = 0
wrong = 0
for x in preddf['predictedvalues']:
    if x == stuff[index][0]:
        correct +=1
    else:
        wrong +=1
    index +=1

print("Correct:", correct)
print("Wrong:", wrong)