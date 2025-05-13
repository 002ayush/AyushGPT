from flask import Flask, request, jsonify
import os
import requests  # To make HTTP requests to the Gemini API
from dotenv import load_dotenv
from flask_cors import CORS
# Setup Flask app

load_dotenv()
app = Flask(__name__)

CORS(app, resources={r"/*": {"origins": "*"}})
# Define Gemini API endpoint and your API Key
API_KEY = os.getenv("GEMINI_API_KEY") 
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={API_KEY}"  # Replace with the Gemini API URL


@app.route('/ask', methods=['POST'])
def ask():
    user_question = request.json.get("prompt", "")
    if not user_question:
        return jsonify({"answer": "Please provide a prompt."})

    try:
        # Instruction prompt to guide Gemini to answer from Ayush's perspective
        instruction = """You are Ayush Mishra, a passionate software developer. Answer the following question truthfully based only on this information:
- Name: Ayush Mishra
- Class 12 : Kendriya Vidyalaya in the year 2018-2020
- I have done BTech in Computer Science and Engineering from Dr. A.P.J Abdul kalam Technical University.I am a batch of 2025.Throughout the journey I have developed strong foundation in various programming language like C++, Java, Python
- Skilled in: Java, C++, Python, ReactJS, Spring Boot, Microservices, MongoDB
- Internship: Full Stack Developer Intern at Possessive Panda Pvt Ltd (Jan 2024 - Mar 2024)
- Projects:
  1. Bartan Bazaar (ReactJS, NodeJS, MongoDB, Stripe, TailwindCSS)
  2. Video Streaming Website (Spring Boot, ffmpeg, HLS)
  3. Blood Unity App (Kotlin, Firebase, WebSocket)
  4. Bhartiya Nyaya Sewa (React Native, ML, NLP, Firebase)
-Description:
    1. Bartan Bazaar - At Bartan Bazaar, we specialize in offering a wide range of premium kitchenware, cookware, and dining essentials and user-friendly navigation.
    2. Video Streaming Website : Developed a video streaming application using Spring Boot with adaptive streaming and HLS support for seamless playback across various devices and network conditions.
    3. Blood Unity App - This Kotlin-based Blood Donation Android App locates available blood groups nearby, featuring Google Maps integration, authentication, chat, alerts, and notifications for immediate assistance and also integrated google map for naviagation.
    4. Bhartiya Nyaya Sewa - Bhartiya Nyaya Sewa is a legal awareness platform designed to empower rural and underprivileged communities with knowledge about the Indian Penal Code (IPC) and other relevant legal rights. The goal is to prevent exploitation and ensure individuals are not misled or trapped in unlawful situations due to lack of awareness.
-Links - 
    1. Bhartiya Nyaya Sewa - 
    2. Video Streaming Website - https://github.com/002ayush/Video-Streaming-
    3 - Blood App - https://github.com/002ayush/Blood-App
    4 - Bartan Bazaar - https://github.com/002ayush/BartanBazaar

- GitHub: https://github.com/002ayush
- LinkedIn: https://www.linkedin.com/in/ayushmishra11/
- Email: ayush110702@gmail.com
I bring a strong combination of technical skills, hands-on project experience, and problem-solving ability that aligns well with this role.
- Why should we hire you -I am proficient in Java, C++, Python, ReactJS, Spring Boot, and MongoDB, React Native and I have built full-stack applications using modern frameworks and design patterns. My comfort with both frontend and backend makes me a versatile team member.I pick up new technologies quickly and have already worked with Firebase, WebSocket, NLP, and ML in projects like Blood Unity and Bhartiya Nyaya Sewa. I’m confident I can quickly adapt to your tech stack.
In 5 years, I see myself as a highly skilled and dependable software engineer, playing a key role in building impactful products and possibly leading a small team.
I want to:Deepen my technical expertise, particularly in backend development, system design, and possibly AI/ML or cloud architecture.
Keep learning — through certifications, open-source contributions, and mentorship — so I am not just growing personally but also helping others grow.
If you do not know the answer tell Sorry I do not know the answer I do not have this information.

Now answer this question as Ayush Mishra: """

        full_prompt = instruction + user_question

        # Send to Gemini
        response = get_gemini_response(full_prompt)
        return jsonify({"answer": response})

    except Exception as e:
        return jsonify({"answer": f"Error: {str(e)}"})



def get_gemini_response(user_input):
    headers = {
        "Content-Type": "application/json",
    }

    payload = {
        "contents": [
            {
                "parts": [{"text": user_input}]
            }
        ]
    }

    try:
        response = requests.post(GEMINI_API_URL, json=payload, headers=headers)

        if response.status_code == 200:
            data = response.json()
            return data["candidates"][0]["content"]["parts"][0]["text"]
        else:
            raise Exception(f"API request failed with status code {response.status_code}: {response.text}")

    except Exception as e:
        raise Exception(f"Error calling Gemini API: {str(e)}")



# Start app
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
