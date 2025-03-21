from flask import Flask, request, jsonify
from chatbot_ln1.application.chat_service import ChatService

app = Flask(__name__)
chat_service = ChatService()

@app.route("/chatbot", methods=["POST"])
def chatbot():
    data = request.json
    mensaje = data.get("message", "")
    respuesta = chat_service.predict(mensaje)
    return jsonify({"response": respuesta})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
