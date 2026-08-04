import pandas as pd

sms_df = pd.read_csv("data/raw/Dataset_5971.csv")
email_df = pd.read_csv("data/raw/phishing_email.csv")

print("=" * 50)
print("SMS DATASET")
print("=" * 50)
print()

print(sms_df.head())
print()

print("Columns:")
print(sms_df.columns)
print()

print("Shape:")
print(sms_df.shape)
print()

print("=" * 50)
print("EMAIL DATASET")
print("=" * 50)
print()

print(email_df.head())
print()

print("Columns:")
print(email_df.columns)
print()

print("Shape:")
print(email_df.shape)