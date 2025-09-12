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

# Contador de tickets por usuario
user_ticket_counter = {}

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
            numero = numero
        return numero
    return None


# ================================
# 🔹 3️⃣ ENDPOINT FLASK
# ================================
@app.route("/chatbot", methods=["POST"])
def chatbot():
    data = request.json
    mensaje = data.get("message", "").strip()
    user_id = data.get("user_id", "default")
    lat = data.get("lat")
    long = data.get("long")

    # Primero, intentamos extraer el teléfono directamente del mensaje
    telefono = extraer_telefono(mensaje)

    if telefono:
        try:
            # Número de ticket iterativo por usuario
            ticket_num = user_ticket_counter.get(user_id, 1)

            # Mensaje que se enviará al número
            mensaje_ticket = f"Se ha registrado su ticket de soporte, número {ticket_num}"

            # Enviar mensaje al API externo
            payload = {
                "phone": "+51986514012",
                "message": mensaje_ticket
            }

            # Si hay latitud/longitud, agregar info de ubicación
            if lat and long:
                google_maps_url = f"https://www.google.com/maps?q={lat},{long}&hl=es-419&markers={lat},{long}"
                payload["message"] += f". Ubicación registrada: {google_maps_url}"

            res = requests.post("http://31.97.11.235:3001/lead", json=payload)
            # print(res.json())
            print(res.status_code)
            # print(f"Enviado a {telefono}: {mensaje_ticket}")
            if res.status_code == 200:
                mensaje_ticket2 = f"Su ticket de soporte se ha Registrado, número {ticket_num}"
                payload2 = {
                    "phone": telefono,
                    "message": mensaje_ticket2
                }

                res2 = requests.post("http://31.97.11.235:3001/lead", json=payload2)
                print(res2.status_code)
                print(res2.json())
            # Incrementar contador para siguiente ticket
            user_ticket_counter[user_id] = ticket_num + 1

            return jsonify({"response": f"Tu ticket #{ticket_num} ha sido registrado. ¡Gracias por contactarnos!", "status": True})

        except Exception as e:
            return jsonify({"response": "Error conectando con el servidor de tickets.", "status": False})

    # Si no es número, buscar respuesta por palabras clave
    mensaje_lower = mensaje.lower()
    respuesta = buscar_respuesta(mensaje_lower)

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
