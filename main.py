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



