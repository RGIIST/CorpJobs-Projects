import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import joblib


df = pd.read_csv('./data/sub_clas.csv')

X = df['sentence']
y = df['subject']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=True)


model_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()), 
    ('classifier', MultinomialNB()) 
])

print('Model traing starts...')
model_pipeline.fit(X_train, y_train)


y_pred = model_pipeline.predict(X_test)

print(classification_report(y_test, y_pred))


joblib.dump(model_pipeline, './models/multinb.joblib')
print('model save!!!')