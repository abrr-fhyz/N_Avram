from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from process import processText
from joblib import dump

def trainModel(features, labels, vectorizer):
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    print("Model trained. Implementing test cases: ")
    
    predictions = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, predictions) * 100)
    print("\n--------------------------------------------------\n")
    return model, vectorizer

def askForSample(model, vectorizer):
    input_text = input("Enter text for sentiment analysis: ")
    
    if input_text.lower() == 'exit':
        print("Exiting the program.")
        return

    if input_text.lower() == 'save':
        print("Saving the model...")
        dump(model, 'data/model.joblib')
        dump(vectorizer, 'data/vectorizer.joblib')
        print("Model Saved")
        return
    
    processed_text = processText(input_text)
    features = vectorizer.transform([processed_text])
    prediction = model.predict(features)
    probability = model.predict_proba(features)
    print("Predicted sentiment:", prediction[0])
    print("Prediction certainity:", probability[0])
    print("\n")
    askForSample(model, vectorizer)


