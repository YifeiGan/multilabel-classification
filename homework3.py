import pandas as pd
import json
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer
import sys
import argparse

# Argument parsing
parser = argparse.ArgumentParser(description='Homework 3: Multi-label Classification of arXiv Paper Summarizations')
parser.add_argument('--data', type=str, required=True, help='Path to the input data file (arxiv_data.json)')
parser.add_argument('--output', type=str, required=True, help='Path to the output results file (results.txt)')
args = parser.parse_args()

train_data_path = args.data
output_path = args.output

# Load data
df = pd.read_json(train_data_path)
df.columns = ['titles', 'summaries', 'labels']

# Encode labels
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(df['labels'])
# Split data
train, valtest = train_test_split(df, test_size=0.30, random_state=1234)
val, test = train_test_split(valtest, test_size=0.50, random_state=1234)
# This will give you a 70/15/15 split. 


# TF-IDF Vectorizer
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words='english')

# Train Multinomial Naive Bayes classifier with OneVsRestClassifier
nb_pipeline = Pipeline([
    ('tfidf', tfidf),
    ('clf', OneVsRestClassifier(MultinomialNB(alpha=0.21500059767808274)))
])

# Fit the model
nb_pipeline.fit(train['summaries'], mlb.transform(train['labels']))

# Predict on validation set
val_pred = nb_pipeline.predict(val['summaries'])

# Calculate and print classification report
val_report = classification_report(mlb.transform(val['labels']), val_pred, target_names=mlb.classes_)

test_pred = nb_pipeline.predict(test['summaries'])
test_report = classification_report(mlb.transform(test['labels']), test_pred, target_names=mlb.classes_)
print("Naive Bayes finished.")

# Train Linear SVC classifier with OneVsRestClassifier
svc_pipeline = Pipeline([
    ('tfidf', tfidf),
    ('clf', OneVsRestClassifier(LinearSVC(C = 2.0324666761898023)))
])

# Fit the model
svc_pipeline.fit(train['summaries'], mlb.transform(train['labels']))

# Predict on validation set
val_pred_svc = svc_pipeline.predict(val['summaries'])

# Calculate and print classification report
val_report_svc = classification_report(mlb.transform(val['labels']), val_pred_svc, target_names=mlb.classes_)

# Predict on test set using the best model (assuming Linear SVC here)
test_pred_svc = svc_pipeline.predict(test['summaries'])

# Calculate and print classification report for test set
test_report_svc = classification_report(mlb.transform(test['labels']), test_pred_svc, target_names=mlb.classes_)
print("Linear SVC finished.")

# Train Logistic Regression classifier with OneVsRestClassifier
lr_pipeline = Pipeline([
    ('tfidf', tfidf),
    ('clf', OneVsRestClassifier(LogisticRegression(max_iter=1000, C = 19.83522113869492)))
])

# Fit the model
lr_pipeline.fit(train['summaries'], mlb.transform(train['labels']))

# Predict on validation set
val_pred_lr = lr_pipeline.predict(val['summaries'])

# Calculate and print classification report
val_report_lr = classification_report(mlb.transform(val['labels']), val_pred_lr, target_names=mlb.classes_)

# Predict on test set using the best model (assuming Logistic Regression here)
test_pred_lr = lr_pipeline.predict(test['summaries'])
test_report_lr = classification_report(mlb.transform(test['labels']), test_pred_lr, target_names=mlb.classes_)
print("Logistic Regression finished.")

# Train Random Forest classifier with OneVsRestClassifier
rf_pipeline = Pipeline([
    ('tfidf', tfidf),
    ('clf', OneVsRestClassifier(RandomForestClassifier()))
])

# Fit the model
rf_pipeline.fit(train['summaries'], mlb.transform(train['labels']))

# Predict on validation set
val_pred_rf = rf_pipeline.predict(val['summaries'])

# Calculate and print classification report
val_report_rf = classification_report(mlb.transform(val['labels']), val_pred_rf, target_names=mlb.classes_)

# Predict on test set using the best model (assuming Random Forest here)
test_pred_rf = rf_pipeline.predict(test['summaries'])
test_report_rf = classification_report(mlb.transform(test['labels']), test_pred_rf, target_names=mlb.classes_)
print("Random Forest finished.")

# Train K-Nearest Neighbors classifier with OneVsRestClassifier
knn_pipeline = Pipeline([
    ('tfidf', tfidf),
    ('clf', OneVsRestClassifier(KNeighborsClassifier()))
])

# Fit the model
knn_pipeline.fit(train['summaries'], mlb.transform(train['labels']))

# Predict on validation set
val_pred_knn = knn_pipeline.predict(val['summaries'])

# Calculate and print classification report
val_report_knn = classification_report(mlb.transform(val['labels']), val_pred_knn, target_names=mlb.classes_)

# Predict on test set using the best model (assuming K-Nearest Neighbors here)
test_pred_knn = knn_pipeline.predict(test['summaries'])
test_report_knn = classification_report(mlb.transform(test['labels']), test_pred_knn, target_names=mlb.classes_)
print("KNN finished.")

# Write validation and test classification reports to txt file
with open(output_path, 'w') as f:
    f.write("Validation Classification Report for Multinomial Naive Bayes:\n")
    f.write(val_report)
    f.write("\n\nTest Classification Report for Multinomial Naive Bayes:\n")
    f.write(test_report)
    
    f.write("\n\nValidation Classification Report for Linear SVC:\n")
    f.write(val_report_svc)
    f.write("\n\nTest Classification Report for Linear SVC:\n")
    f.write(test_report_svc)
    
    f.write("\n\nValidation Classification Report for Logistic Regression:\n")
    f.write(val_report_lr)
    f.write("\n\nTest Classification Report for Logistic Regression:\n")
    f.write(test_report_lr)
    
    f.write("\n\nValidation Classification Report for Random Forest:\n")
    f.write(val_report_rf)
    f.write("\n\nTest Classification Report for Random Forest:\n")
    f.write(test_report_rf)
    
    f.write("\n\nValidation Classification Report for K-Nearest Neighbors:\n")
    f.write(val_report_knn)
    f.write("\n\nTest Classification Report for K-Nearest Neighbors:\n")
    f.write(test_report_knn)

print("output file generated.")