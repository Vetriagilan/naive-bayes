# 📩 SMS Spam Detection App

A simple **Streamlit web app** that uses a **Naive Bayes text classifier** to detect whether an SMS message is **Spam** or **Ham (Not Spam)**.

---

## 🚀 Features
- Loads and preprocesses the **SMS Spam Collection Dataset** (`spam.csv`)
- Balances the dataset using **oversampling**
- Converts text to numerical features using **CountVectorizer**
- Trains a **Multinomial Naive Bayes** classifier
- Interactive web interface built with **Streamlit**
- Displays prediction and confidence score instantly

---

## 🧠 How It Works
1. **Data Preparation**
   - Reads `spam.csv` (columns: `v1` = label, `v2` = message)
   - Renames columns to `label` and `message`
   - Maps labels (`ham → 0`, `spam → 1`)
   - Balances the dataset using random oversampling

2. **Model Training**
   - Splits data into train/test sets (80/20)
   - Converts text into word frequency vectors using **CountVectorizer**
   - Trains a **Multinomial Naive Bayes** model on the messages

3. **Prediction**
   - User inputs an SMS message
   - The message is vectorized and passed to the trained model
   - The app predicts whether it is **Spam** or **Ham**

---

## 🧩 File Structure

## Streamlit app: https://naive-bayes-endbsyutdappc6kv8hylsxh.streamlit.app/