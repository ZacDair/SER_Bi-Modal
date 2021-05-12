# BET EMOTIONS [Fear, Anger, Joy, Sadness, Disgust, Surprise]
import text2emotion as te  # Emotion Range: [Fear, Angry, Happy, Sad, Surprise]
import pandas as pd

# Changes Pandas display options to support the large strings
pd.options.display.max_colwidth = 200


# Helper function to get the key or keys corresponding to a value
def getDictKeyByVal(dictToSearch, val):
    result = []
    for key in dictToSearch:
        if dictToSearch[key] == val:
            result.append(key)
    return result


# Run Model text2emotion
def runText2Emotion(data):
    results = []
    c = 0
    w = 0
    i = 0
    while i < len(data):
        entry = data.iloc[i]
        pred = te.get_emotion(entry["text"])
        foundEmotions = getDictKeyByVal(pred, max(pred.values()))
        if entry["label"] in foundEmotions:
            c +=1
        else:
            w +=1

        results.append([pred, entry["label"]])
        i += 1

    return results, c, w


# Read in the dataset
data = pd.read_csv("../../Datasets/ISEAR.csv")  # Emotion Range: [joy, fear, anger, sadness, disgust, shame, guilt]

# Remove \n chars
data["text"] = data["text"].str.replace("\n", "")

# Drop emotions that aren't supported by text2emotion
testableData = data[data["label"] != "disgust"]
testableData = testableData[testableData["label"] != "shame"]
testableData = testableData[testableData["label"] != "guilt"]

# Alter labels to match text2emotion

testableData["label"] = testableData["label"].str.replace("joy", "Happy")
testableData["label"] = testableData["label"].str.replace("fear", "Fear")
testableData["label"] = testableData["label"].str.replace("anger", "Angry")
testableData["label"] = testableData["label"].str.replace("sadness", "Sad")
# print(testableData)

res, correct, wrong = runText2Emotion(testableData)
print("Text2Speech:")
print("Correct:", correct)
print("Wrong:", wrong)
print("Accuracy:", correct/(correct+wrong))
print("\nLabel Vs Predictions:")
for val in res:
    print(val)




