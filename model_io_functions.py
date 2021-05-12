import os
import pickle
from keras.models import model_from_json


# Validate the presence of the audio file
def checkForFile(filepath):
    if os.path.exists(filepath):
        return True
    else:
        return False


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


# Load the models from pickled files
def loadPickledModel(filepath):
    try:
        loaded_model = pickle.load(open(filepath, 'rb'))
        return loaded_model
    except FileNotFoundError:
        print("ERROR - The imported model files were not found")
        exit(1)


# Load model and weights from Json
def loadJsonModel(weightFile, modelFile):
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