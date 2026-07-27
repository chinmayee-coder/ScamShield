from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# -----------------------------
# 1. Load dataset
# -----------------------------

data = pd.read_csv("spam.csv")

x = data["Message"]

y = data["spamORham"]


# -----------------------------
# 2. Convert labels
# -----------------------------

y = y.map({
    "ham": 0,
    "spam": 1
})


# -----------------------------
# 3. Split dataset
# -----------------------------

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42
)


# -----------------------------
# 4. TF-IDF
# -----------------------------

vectorizer = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2)
)

x_train_vector = vectorizer.fit_transform(x_train)

x_test_vector = vectorizer.transform(x_test)


# Convert sparse matrices to dense arrays
x_train_vector = x_train_vector.toarray()

x_test_vector = x_test_vector.toarray()


# -----------------------------
# 5. ANN Model
# -----------------------------

model = Sequential([
    Dense(32, activation="relu"),
    Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)


# -----------------------------
# 6. Train model
# -----------------------------

history = model.fit(
    x_train_vector,
    y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.2
)


# -----------------------------
# 7. Predictions
# -----------------------------

predictions = model.predict(x_test_vector)

predictions = (predictions > 0.5).astype(int)


# -----------------------------
# 8. Evaluation
# -----------------------------

cm = confusion_matrix(
    y_test,
    predictions
)

print("Confusion Matrix:")
print(cm)

print("\nClassification Report:")
print(classification_report(
    y_test,
    predictions
))


precision = precision_score(
    y_test,
    predictions,
    pos_label=1
)

recall = recall_score(
    y_test,
    predictions,
    pos_label=1
)

f1 = f1_score(
    y_test,
    predictions,
    pos_label=1
)

accuracy = accuracy_score(
    y_test,
    predictions
)

print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("Accuracy:", accuracy)


# -----------------------------
# 9. Test custom message
# -----------------------------

new_message = [
    "claim your prize now!"
]

new_message_vector = vectorizer.transform(
    new_message
)

new_message_vector = new_message_vector.toarray()

prediction = model.predict(
    new_message_vector
)

spam_probability = prediction[0][0]

print("Spam Probability:", spam_probability)

if spam_probability > 0.5:
    print("Prediction: SPAM")
else:
    print("Prediction: HAM")