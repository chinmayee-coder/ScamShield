from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
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
vectorizer=CountVectorizer()
x_train_vector=vectorizer.fit_transform(x_train)
model=MultinomialNB()
model.fit(x_train_vector,y_train)
x_test_vector=vectorizer.transform(x_test)
predictions=model.predict(x_test_vector)
print(predictions[:10])
accuracy=accuracy_score(y_test,predictions)
print("Accuracy:",accuracy)
new_message=["claim your prize now!"]
new_message_vector=vectorizer.transform(new_message)
prediction=model.predict(new_message_vector)
print("Prediction:",prediction[0])
