from nltk.stem import PorterStemmer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
data=pd.read_csv("spam.csv")
x=data["Message"]
y=data["spamORham"]
x_train,x_test,y_train,y_test=train_test_split(
  x,
  y,
  test_size=0.2,
  random_state=42
)
print("Training message:",len(x_train))
print("Testing message:",len(x_test))
vectorizer=TfidfVectorizer(stop_words="english")
x_train_vector=vectorizer.fit_transform(x_train)
model=LogisticRegression(max_iter=1000)
model.fit(x_train_vector,y_train)
x_test_vector=vectorizer.transform(x_test)
predictions=model.predict(x_test_vector)
cm=confusion_matrix(y_test, predictions)
print(cm)
precision=precision_score(
  y_test,
  predictions,
  pos_label="spam"
)
recall=recall_score(
  y_test,
  predictions,
  pos_label="spam"
)
print(classification_report(y_test,predictions))
print("Precision:",precision)
print("Recall:", recall)
f1=f1_score(
  y_test,
  predictions,
  pos_label="spam"
)
print("F1 Score:",f1)
print(predictions[:10])
accuracy=accuracy_score(y_test,predictions)
print("Accuracy:",accuracy)
new_message=["claim your prize now!"]
new_message_vector=vectorizer.transform(new_message)
prediction=model.predict(new_message_vector)
print("Prediction:",prediction[0])
feature_names=vectorizer.get_feature_names_out()
print(feature_names[:20])
print(model.classes_)
weights=model.coef_[0]
#spam_words=model.feature_log_prob_[1]
top_spam_indices=weights.argsort()[-10:]
print("Top spam words")
for i in top_spam_indices:
    print(feature_names[i])
