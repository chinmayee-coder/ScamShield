import pandas as pd

# -----------------------------
# Load SMS Dataset
# -----------------------------
sms_df = pd.read_csv("data/raw/Dataset_5971.csv")

# -----------------------------
# Load Email Dataset
# -----------------------------
email_df = pd.read_csv("data/raw/phishing_email.csv")
# -----------------------------
# Rename SMS columns
# -----------------------------
sms_df = sms_df.rename(
    columns={
        "TEXT": "text",
        "LABEL": "label"
    }
)
print(sms_df.head())
# -----------------------------
# Rename Email columns
# -----------------------------
email_df = email_df.rename(
    columns={
        "text_combined": "text"
    }
)

print(email_df.head())
# -----------------------------
# Standardize SMS labels
# -----------------------------
sms_df["label"] = sms_df["label"].replace({
    "Smishing": "spam",
    "smishing": "spam",
    "Spam": "spam",
    "spam": "spam",
    "ham": "ham"
})

print("\nSMS Labels:")
print(sms_df["label"].value_counts())
print("\nEmail Labels:")
print(email_df["label"].value_counts())
print("\nOne email with label 0")
print(email_df[email_df["label"] == 0]["text"].iloc[0])

print("\n" + "=" * 80)

print("\nOne email with label 1")
print(email_df[email_df["label"] == 1]["text"].iloc[0])
# -----------------------------
# Standardize Email labels
# -----------------------------
email_df["label"] = email_df["label"].replace({
    0: "ham",
    1: "spam"
})

print("\nEmail Labels After Standardization:")
print(email_df["label"].value_counts())
# -----------------------------
# Load Original SMS Spam Dataset
# -----------------------------
old_sms_df = pd.read_csv("data/raw/spam.csv", encoding="latin-1")
print("\nOriginal SMS Dataset")
print(old_sms_df.head())

print("\nColumns:")
print(old_sms_df.columns)
# -----------------------------
# Clean Original SMS Dataset
# -----------------------------
old_sms_df = old_sms_df.drop(columns=["Unnamed: 0"])

old_sms_df = old_sms_df.rename(
    columns={
        "spamORham": "label",
        "Message": "text"
    }
)

print("\nOriginal SMS After Cleaning")
print(old_sms_df.head())
# -----------------------------
# Merge all datasets
# -----------------------------
final_df = pd.concat(
    [old_sms_df, sms_df[["text", "label"]], email_df],
    ignore_index=True
)

print("\nFinal Dataset Shape:")
print(final_df.shape)

print("\nFirst 5 Rows:")
print(final_df.head())
# -----------------------------
# Remove Duplicate Messages
# -----------------------------
print("\nDataset Shape Before Removing Duplicates:")
print(final_df.shape)

final_df = final_df.drop_duplicates(subset=["text"])

print("\nDataset Shape After Removing Duplicates:")
print(final_df.shape)
# -----------------------------
# Save Final Dataset
# -----------------------------
final_df.to_csv(
    "data/processed/final_dataset.csv",
    index=False
)

print("\nFinal dataset saved successfully!")