import pickle

from keras.models import model_from_json
import pandas as pd
import os
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten, Dropout, Activation
from keras.layers import Conv1D, MaxPooling1D
from tensorflow.keras import optimizers
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from keras.utils import np_utils


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


# Save the model and weights
def saveModel(modelName, model):
    save_dir = os.path.join(os.getcwd(), 'my_models')
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    model_path = os.path.join(save_dir, modelName+".h5")
    model.save(model_path)
    print('Saved trained model at %s ' % model_path)
    model_json = model.to_json()
    with open(save_dir + "/" + modelName+".json", "w") as json_file:
        json_file.write(model_json)


# Save the model and weights
def savePickleModel(modelName, model):
    save_dir = os.path.join(os.getcwd(), 'my_models')
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    model_path = os.path.join(save_dir, modelName+".sav")
    pickle.dump(model, open(model_path, 'wb'))
    print('Saved trained model at %s ' % model_path)


# Load the dataset
dataset = pd.read_csv("../Datasets/RAVDESS/speechActorDataset2.csv")

# Split features and labels
datasetCopy = dataset
labels = datasetCopy['label']
features = datasetCopy.drop(columns='label')

# Train Test Split
trainIndex = int(len(features) * 0.7)
train_features = features[:trainIndex]
train_labels = labels[:trainIndex]
test_features = features[trainIndex+1:-1]
test_labels = labels[trainIndex+1:-1]

lb = LabelEncoder()
y_train = np_utils.to_categorical(lb.fit_transform(train_labels))
savePickleModel("baseline_AP_labelEncoder", lb)
y_test = np_utils.to_categorical(lb.fit_transform(test_labels))

# Changing Dimension for CNN model
x_traincnn =np.expand_dims(train_features, axis=2)
x_testcnn= np.expand_dims(test_features, axis=2)

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
model.add(Dense(5))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',optimizer="SGD",metrics=['accuracy'])
cnnhistory=model.fit(x_traincnn, y_train, batch_size=16, epochs=300, validation_data=(x_testcnn, y_test))

preds = model.predict(x_testcnn, batch_size=16, verbose=1)
preds1=preds.argmax(axis=1)
abc = preds1.astype(int).flatten()
predictions = (lb.inverse_transform((abc)))
preddf = pd.DataFrame({'predictedvalues': predictions})

plt.plot(cnnhistory.history['accuracy'])
plt.plot(cnnhistory.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

saveModel("affectiveProsody_model", model)