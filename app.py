import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

# -------------------------------
# 📦 Load and prepare data
# -------------------------------
@st.cache_resource
def load_model():
    df = pd.read_csv("spam.csv", encoding="latin-1")
    df = df.rename(columns={"v1": "label", "v2": "message"})
    df = df[["label", "message"]]
    df["label"] = df["label"].map({"ham": 0, "spam": 1})

    # Balance dataset
    df_ham = df[df["label"] == 0]
    df_spam = df[df["label"] == 1]
    df_spam_up = resample(df_spam,
                          replace=True,
                          n_samples=len(df_ham),
                          random_state=42)
    df_balanced = pd.concat([df_ham, df_spam_up])

    # Train model
    X_train, X_test, y_train, y_test = train_test_split(
        df_balanced["message"], df_balanced["label"],
        test_size=0.2, random_state=42
    )
    vectorizer = CountVectorizer(stop_words='english')
    X_train_vec = vectorizer.fit_transform(X_train)

    model = MultinomialNB()
    model.fit(X_train_vec, y_train)

    return model, vectorizer

model, vectorizer = load_model()

# -------------------------------
# 🖥️ Streamlit UI
# -------------------------------
st.title("📩 SMS Spam Detection App")
st.write("Enter a message below and find out whether it's **Spam** or **Ham (Not Spam)**.")

# Input box
user_msg = st.text_area("✉️ Enter your message here:")

if st.button("Predict"):
    if user_msg.strip() == "":
        st.warning("Please enter a message.")
    else:
        # Transform input message
        msg_vec = vectorizer.transform([user_msg])
        prediction = model.predict(msg_vec)[0]
        prob = model.predict_proba(msg_vec)[0][prediction]

        if prediction == 1:
            st.error(f"🚨 SPAM detected! (Confidence: {prob:.2%})")
        else:
            st.success(f"✅ HAM (Not Spam). (Confidence: {prob:.2%})")

# Optional footer
st.markdown("---")
st.caption("Built with ❤️ using Streamlit and Naive Bayes.")
