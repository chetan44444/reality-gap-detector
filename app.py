import streamlit as st
import pickle

# Load model
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

st.title("🧠 Reality Gap Detector (AI)")
st.write("Analyze whether your statement is realistic or not")

user_input = st.text_area("Enter your claim:")

# Keywords for explanation
unrealistic_keywords = [
    "without doing anything",
    "no work",
    "no effort",
    "instantly",
    "overnight",
    "easy money",
    "quick success"
]

if st.button("Analyze"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text")
    else:
        text = user_input.lower()

        # 🔴 Rule-based check
        found_words = []
        for word in unrealistic_keywords:
            if word in text:
                found_words.append(word)

        if found_words:
            score = 20
            st.error("❌ Unrealistic (Rule-based detection 🚨)")
            st.write("⚠️ Reason: Found unrealistic phrases →", ", ".join(found_words))
        
        else:
            # 🤖 ML Prediction
            data = vectorizer.transform([user_input])
            prediction = model.predict(data)[0]
            confidence = model.predict_proba(data).max()

            if prediction == 0:
                score = int(confidence * 100)
                st.success("✅ Reality aligned")
            else:
                score = int((1 - confidence) * 100)
                st.error("❌ Unrealistic")

        # 🎯 Score display
        st.subheader(f"Reality Score: {score}/100")
        st.progress(score)

        # 🧠 Insight
        if score > 70:
            st.info("💡 Strong realistic thinking")
        elif score > 40:
            st.info("⚖️ Moderate realism")
        else:
            st.info("🚨 High reality gap detected")