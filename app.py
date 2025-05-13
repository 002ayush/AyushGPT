from flask import Flask, request, jsonify
# from flask_ngrok import run_with_ngrok
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
# from pyngrok import ngrok
import os
import pickle

# 🔃 Load facts and embeddings from pickle
with open("data.pkl", "rb") as f:
    data = pickle.load(f)
    facts = data["facts"]
    print("facts", facts)
    fact_embeddings = data["embeddings"]

# Load model (make sure same model used to generate embeddings)
model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')




# Setup Flask app
app = Flask(__name__)
# run_with_ngrok(app)

@app.route('/ask', methods=['POST'])
def ask():
    prompt = request.json.get("prompt", "")
    if not prompt:
        return jsonify({"answer": "Please provide a prompt."})

    sub_questions = [q.strip() for q in prompt.lower().split(" and ") if q.strip()]
    final_answers = []

    for sub_q in sub_questions:
        if "all projects" in sub_q or ("projects" in sub_q and "all" in sub_q) or "list of projects" in sub_q:
            project_facts = [fact for fact in facts if "project" in fact.lower() or "description of" in fact.lower()]
            if project_facts:
                final_answers.append("<br><br>".join(project_facts))
            else:
                final_answers.append("No project details found.")
            continue

        query_embedding = model.encode([sub_q])
        scores = cosine_similarity(query_embedding, fact_embeddings)[0]
        best_index = scores.argmax()
        best_score = scores[best_index]

        if best_score < 0.2:
            final_answers.append("Sorry, I don't know the answer to that question.")
        else:
            final_answers.append(facts[best_index])

    return jsonify({"answer": "<br><br>".join(final_answers)})

# Start app
# public_url = ngrok.connect(5000)
# print(f" * ngrok tunnel: {public_url}")
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
