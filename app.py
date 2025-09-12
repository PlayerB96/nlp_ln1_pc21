from flask import Flask, request, jsonify
import requests
import re

app = Flask(__name__)

# ==================================
# 🔹 1️⃣ PALABRAS CLAVE Y RESPUESTAS
# ==================================
chat_responses = [
    {"id": 1, "response": "¡Hola! ¿En qué puedo ayudarte hoy?"},
    {
        "id": 2,
        "response": "Claro, ofrecemos servicios de soporte técnico, desarrollo web y consultoría, escribe la palabra 'ticket' para darte atención.",
    },
    {
        "id": 3,
        "response": "Nuestro horario de atención es de lunes a viernes, de 9am a 6pm.",
    },
    {
        "id": 4,
        "response": "Por favor, envíame tu número de teléfono para ayudarte con tu ticket.",
    },
    {"id": 5, "response": "Tu ticket ha sido registrado. ¡Gracias por contactarnos!"},
]

chat_keywords = [
    {"keyword": "hola", "chat_response_id": 1},
    {"keyword": "buenos días", "chat_response_id": 1},
    {"keyword": "servicio", "chat_response_id": 2},
    {"keyword": "soporte", "chat_response_id": 2},
    {"keyword": "desarrollo", "chat_response_id": 2},
    {"keyword": "horario", "chat_response_id": 3},
    {"keyword": "atención", "chat_response_id": 3},
    {"keyword": "ticket", "chat_response_id": 4},
]

# Estados de los usuarios
user_states = {}


# ================================
# 🔹 2️⃣ FUNCIONES AUXILIARES
# ================================
def buscar_respuesta(mensaje: str):
    """
    Busca una respuesta en base a coincidencia de palabra clave.
    """
    for ck in chat_keywords:
        if ck["keyword"] in mensaje:
            respuesta = next(
                (cr["response"] for cr in chat_responses if cr["id"] == ck["chat_response_id"]),
                None,
            )
            return respuesta
    return None


def extraer_telefono(texto):
    match = re.search(r"(\+51)?9\d{8}", texto)
    if match:
        numero = match.group(0)
        if not numero.startswith("+51"):
            numero =  numero
        return numero
    return None


# ================================
# 🔹 3️⃣ ENDPOINT FLASK
# ================================
@app.route("/chatbot", methods=["POST"])
def chatbot():
    data = request.json
    mensaje = data.get("message", "").lower()
    user_id = data.get("user_id", "default")
    lat = data.get("lat")
    long = data.get("long")

    # 1️⃣ Estado esperando número de teléfono
    if user_states.get(user_id) == "esperando_telefono":
        telefono = extraer_telefono(mensaje)
        if telefono:
            try:
                # Primer mensaje: validación
                res1 = requests.post(
                    "http://31.97.11.235:3001/lead",
                    json={"phone": telefono, "message": "VALIDACION DE TICKET"},
                )
                print(res1.json())
                print(telefono)
                if lat and long:
                    google_maps_url = f"https://www.google.com/maps?q={lat},{long}&hl=es-419&markers={lat},{long}"
                    res2 = requests.post(
                        "http://31.97.11.235:3001/lead",
                        json={
                            "phone": telefono,
                            "message": f"Se generó un ticket de soporte desde la ubicación: {google_maps_url}",
                        },
                    )
                    print(res2.json())
                    if res1.status_code == 200 and res2.status_code == 200:
                        user_states.pop(user_id, None)
                        return jsonify({"response": chat_responses[4]["response"], "status": True})
                    else:
                        user_states.pop(user_id, None)
                        return jsonify(
                            {"response": "Hubo un problema al registrar tu ticket. Intenta más tarde.", "status": False}
                        )
                else:
                    res2 = requests.post(
                        "http://31.97.11.235:3001/lead",
                        json={
                            "phone": telefono,
                            "message": "Se generó un ticket de soporte, pero no se recibió información de ubicación.",
                        },
                    )
                    user_states.pop(user_id, None)
                    return jsonify(
                        {"response": "Tu ticket fue registrado, pero no se recibió ubicación.", "status": False}
                    )
            except Exception as e:
                user_states.pop(user_id, None)
                return jsonify({"response": "Error conectando con el servidor de tickets.", "status": False})
        else:
            user_states.pop(user_id, None)
            return jsonify(
                {"response": "No pude detectar tu número. Asegúrate de enviarlo en formato 9XXXXXXXX o +519XXXXXXXX. Iniciemos de nuevo.", "status": False}
            )

    # 2️⃣ Búsqueda de respuesta por palabras clave
    respuesta = buscar_respuesta(mensaje)

    # Si el usuario dijo "ticket", guardamos el estado
    if respuesta == chat_responses[3]["response"]:
        user_states[user_id] = "esperando_telefono"

    if respuesta:
        return jsonify({"response": respuesta, "status": False})
    else:
        return jsonify({"response": "Lo siento, no entendí eso.", "status": False})


# ================================
# 🔹 4️⃣ CORRER LA APP
# ================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=6000, debug=True)
