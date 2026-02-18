from flask import Flask, render_template, request
import nltk
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt')

app = Flask(__name__)

# -------------------------
# Step 1: Collect FAQs
# -------------------------
faqs = {
    "What is AI?": "Artificial Intelligence is the simulation of human intelligence in machines.",
    "What is Machine Learning?": "Machine Learning is a subset of AI that learns from data.",
    "What is NLP?": "NLP stands for Natural Language Processing.",
    "How does this chatbot work?": "It matches user questions with FAQs using cosine similarity.",
    "Who created this chatbot?": "This chatbot was created as an academic mini project."
}

questions = list(faqs.keys())
answers = list(faqs.values())

# -------------------------
# Step 2: Preprocessing
# -------------------------
def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

processed_questions = [preprocess(q) for q in questions]

# -------------------------
# Step 3: Vectorization
# -------------------------
vectorizer = TfidfVectorizer()
faq_vectors = vectorizer.fit_transform(processed_questions)

# -------------------------
# Step 4: Matching Logic
# -------------------------
def get_answer(user_input):
    user_input = preprocess(user_input)
    user_vector = vectorizer.transform([user_input])
    similarity = cosine_similarity(user_vector, faq_vectors)
    best_match = similarity.argmax()
    score = similarity[0][best_match]

    if score < 0.3:
        return "Sorry, I don't understand your question."
    return answers[best_match]

# -------------------------
# Step 5: Routes
# -------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    response = ""
    if request.method == "POST":
        user_input = request.form["question"]
        response = get_answer(user_input)
    return render_template("index.html", response=response)

if __name__ == "__main__":
    app.run(debug=True)
