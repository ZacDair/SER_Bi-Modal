# BET EMOTIONS [Fear, Anger, Joy, Sadness, Disgust, Surprise]
import nltk
import pandas as pd

# Changes Pandas display options to support the large strings
pd.options.display.max_colwidth = 200

# Read in the emotional text dataset
data = pd.read_csv("../../Datasets/ISEAR.csv") # Emotion Range: [joy, fear, anger, sadness, disgust, shame, guilt]

# Remove \n chars
data["text"] = data["text"].str.replace("\n", "")
data["text"] = data["text"].str.replace(".", "")

# Drop emotions we don't use
testableData = data[data["label"] != "shame"]
testableData = testableData[testableData["label"] != "guilt"]


# Read in the emotional lexicon
filepath = "../../Datasets/NRC_emotion_lexicon_list.txt"
emolex_df = pd.read_csv(filepath,  names=["word", "emotion", "association"], skiprows=45, sep='\t')
emolex_df.head(12)

# Remove all emotions we don't use (anticipation, negative, positive, trust, anticipation)
# or only use the emotions we need
anger = emolex_df[(emolex_df.association == 1) & (emolex_df.emotion == 'anger')].word
joy = emolex_df[(emolex_df.association == 1) & (emolex_df.emotion == 'joy')].word
fear = emolex_df[(emolex_df.association == 1) & (emolex_df.emotion == 'fear')].word
sadness = emolex_df[(emolex_df.association == 1) & (emolex_df.emotion == 'sadness')].word
disgust = emolex_df[(emolex_df.association == 1) & (emolex_df.emotion == 'disgust')].word
surprise = emolex_df[(emolex_df.association == 1) & (emolex_df.emotion == 'surprise')].word
print(len(anger))
print(len(fear))
print(len(joy))
print(len(sadness))
print(len(disgust))
print(len(surprise))

keys = ["anger", "joy", "fear", "sadness", "disgust"]
index = 0
correct = 0
wrong = 0
for x in testableData["text"]:
    commonWords = {}
    words = nltk.word_tokenize(x)
    commonWords["anger"] = len(set(words) & set(anger))
    commonWords["joy"] = len(set(words) & set(joy))
    commonWords["fear"] = len(set(words) & set(fear))
    commonWords["sadness"] = len(set(words) & set(sadness))
    commonWords["disgust"] = len(set(words) & set(disgust))
    commonWords["count"] = len(words)
    result = {}
    for key in keys:
        if commonWords[key] != 0:
            result[key] = (commonWords[key] / commonWords["count"]) * 100

    if testableData.iloc[index]["label"] in result:
        correct += 1
    else:
        wrong += 1

    print("My guess: ", result)
    print("Label: ", testableData.iloc[index]["label"])
    index += 1

print("Overall results:")
print(correct, "Correct")
print(wrong, "Wrong")
print("Accuracy: ", correct/(correct+wrong))
