import tkinter as tk
from tkinter import messagebox
import joblib
from process import processText

model = joblib.load('data/model_1.joblib')
vectorizer = joblib.load('data/vectorizer_1.joblib')

def update_label(prediction):
    if prediction == 'positive':
        prediction_label.config(text="Positive", fg="green")
    else:
        prediction_label.config(text="Negative", fg="red")

def update_confidence(probability):
    confidence_score = max(probability) * 100
    if confidence_score < 60 :
        confidence_label.config(text="Low", fg="black")
    elif confidence_score < 70 :
        confidence_label.config(text="Medium", fg="orange")
    else :
        confidence_label.config(text="High", fg="green")

def predict_sentiment():
    input_text = text_entry.get("1.0", tk.END).strip()
    
    if not input_text:
        messagebox.showwarning("Input Error", "Please enter some text to analyze.")
        return
    input_text = processText(input_text)
    input_features = vectorizer.transform([input_text])
    
    prediction = model.predict(input_features)[0]
    probability = model.predict_proba(input_features)[0]
    update_label(prediction)
    update_confidence(probability)

window = tk.Tk()
window.title("N_Avram Sentiment Analysis Tool")
window.geometry("600x400")

input_frame = tk.Frame(window, bg="#f2f2f2")
input_frame.pack(pady=15)

instruction_label = tk.Label(input_frame, text="Enter review for sentiment analysis:", font=("Arial", 15, "bold"), bg="#f2f2f2")
instruction_label.pack()

text_entry = tk.Text(window, height=10, width=60, wrap="word")
text_entry.pack(pady=5)

predict_button = tk.Button(window, text="Predict Sentiment", font=("Arial", 12, "bold"), command=predict_sentiment, bg="#00239c", fg="white", padx=10, pady=5)
predict_button.pack(pady=10)

result_frame = tk.Frame(window, bg="#f2f0ef")
result_frame.pack(pady=10)

result_label = tk.Label(result_frame, text="Prediction:", font=("Arial", 12, "bold"), bg="#f2f2f2")
result_label.grid(row=0, column=0, sticky="e")

prediction_label = tk.Label(result_frame, text="N/A", font=("Arial", 12), bg="#f2f2f2", fg="#333333")
prediction_label.grid(row=0, column=1, padx=10)

confidence_text_label = tk.Label(result_frame, text="Confidence:", font=("Arial", 12, "bold"), bg="#f2f2f2")
confidence_text_label.grid(row=1, column=0, sticky="e")

confidence_label = tk.Label(result_frame, text="N/A", font=("Arial", 12), bg="#f2f2f2", fg="#333333")
confidence_label.grid(row=1, column=1, padx=10)


window.mainloop()
