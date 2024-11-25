# **Sentiment Analysis Project**

---

## **Overview**
This project implements a sentiment analysis tool using a `RandomForestClassifier`. It processes user reviews to predict sentiment (positive or negative) and outputs confidence levels. A GUI is included for interactive use, alongside a training script.

---

## **Working Process**
1. **Preprocessing**: 
   - Removes stop-words, lemmatizes text, and uses TF-IDF vectorization with n-grams (1 to 3) for feature extraction.
2. **Training**:
   - Trains a `RandomForestClassifier` using labeled datasets.
3. **Prediction**:
   - Processes user input, vectorizes it, and predicts sentiment with confidence scores.

---

## **Features**
- **Custom Preprocessing**: Includes advanced text cleaning, lemmatization, and n-gram analysis.
- **Polarity Features**: Adds counts of positive and negative words for enhanced feature representation.
- **User-Friendly GUI**: An interactive tool for real-time sentiment predictions.
- **Optimized Inference**: High-speed predictions suitable for live applications.

---

## **Observed Results**
- **Accuracy**: ~83.5%.
- **Recall**: 91.3% (excellent detection of positive sentiments).
- **Precision**: 83.5% (reliable positive review classification).
- **Inference Time**: ~0.00085 seconds per prediction.

---

## **Requirements**
Install the following libraries to run the project:
```bash
pip install pandas numpy scikit-learn nltk joblib tkinter
```
# **How to Run**
Train the Model: Run the following command to train the model:
```bash
python main.py
```
Run the GUI: Launch the GUI to test individual reviews:
```bash
python runModel.py
```

## **DataSet**
https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews