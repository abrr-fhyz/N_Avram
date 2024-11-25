import pandas as pd
import joblib
import time
from sklearn.metrics import confusion_matrix, recall_score, precision_score, log_loss
from sklearn.metrics import roc_curve, accuracy_score

def testing(model, X_test, y_test):
	print("Performing Metrics check...")
	start_time = time.time()
	y_pred = model.predict(X_test)
	end_time = time.time()
	inference_time = (end_time - start_time) / len(y_test)
	print("Done. Analyzing Results...")
	y_pred_proba = model.predict_proba(X_test)

	conf_matrix = confusion_matrix(y_test, y_pred)
	precision = precision_score(y_test, y_pred, pos_label='positive')
	recall = recall_score(y_test, y_pred, pos_label='positive')
	log_loss_value = log_loss(y_test, y_pred_proba)

	tn, fp, fn, tp = conf_matrix.ravel()
	fpr = fp / (fp + tn)  
	fnr = fn / (fn + tp)  

	print("Confusion Matrix:\n", conf_matrix)
	print("Precision:", precision)	
	print("Recall:", recall)
	print("Log Loss:", log_loss_value)
	print("False Positive Rate (FPR):", fpr)
	print("False Negative Rate (FNR):", fnr)
	print("Inference Time per Prediction (seconds):", inference_time)

print("Loading model...")
model = joblib.load('data/model.joblib')
vectorizer = joblib.load('data/vectorizer.joblib')
print("Model Loaded.")
data = pd.read_csv('data/dataset.csv')

print("Processing Data...")
X_test = vectorizer.transform(data['review'])
y_test = data['sentiment']
print("Data Processed.")

testing(model, X_test, y_test)

