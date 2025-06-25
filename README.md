# 🎬 Sentiment Analysis Web App (IMDB Movie Reviews)

A machine learning web app to classify IMDB movie reviews as **Positive** or **Negative**, using TF-IDF + Linear SVM and deployed via **Gradio**.

Built as part of my internship project at **AccelerateX**.

---

## 🌐 Live Demo

▶️ **Try it here**:  
👉 [Hugging Face Space](https://huggingface.co/spaces/suhana2004/movie-review-sentiment)

---

## 🔗 Download Model Files

Due to GitHub file size limits, the model files are hosted on Hugging Face:

- 📦 [Download `best_model_linear_svm.pkl`](https://huggingface.co/spaces/suhana2004/movie-review-sentiment/blob/main/best_model_linear_svm.pkl)
- 📦 [Download `tfidf_vectorizer.pkl`](https://huggingface.co/spaces/suhana2004/movie-review-sentiment/blob/main/tfidf_vectorizer.pkl)

👉 Place both files in the **same folder** as `app.py`.

---

## 🧠 Model Info

- **Vectorizer**: TF-IDF (bigrams)
- **Classifier**: Linear SVM (`LinearSVC`)
- **Dataset**: IMDB Movie Reviews (50k labeled reviews)
- **Test Accuracy**: ~87%

---

## 💻 How to Run Locally

```bash
# Clone this repo
git clone https://github.com/your-username/sentiment-analysis-app.git
cd sentiment-analysis-app

# Install required packages
pip install -r requirements.txt

# Run the app
python app.py
