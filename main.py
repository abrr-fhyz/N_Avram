import pandas as pd
from tqdm import tqdm
from process import processText, featureExtraction, featureExtractionSvd
from train import trainModel, askForSample

print("Loading data...")
data = pd.read_csv('data/smallerData.csv')

print("Processing data...")
tqdm.pandas(desc="Processing Reviews")
data['processed_text'] = data['review'].progress_apply(processText)

print("Implementing feature extraction:")
print("Model_02...")
features_2, labels_2, vectorizer_2 = featureExtraction(data, ngram_range=(1, 3)) 
#print("Model_08...") # best so far
#features_8, labels_8, vectorizer_8 = featureExtraction(data, ngram_range=(1, 6))


print("Training models:")
print("Model_02...")
model_2, vectorizer_2 = trainModel(features_2, labels_2, vectorizer_2)
#print("Model_08... Best Model So Far")
#model_8, vectorizer_8 = trainModel(features_8, labels_8, vectorizer_8)

askForSample(model_2, vectorizer_2)

