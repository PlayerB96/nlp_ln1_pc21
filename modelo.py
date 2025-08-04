import spacy
from spacy.training.example import Example
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

# Unir datos simulados
train_data = []
for ck in chat_keywords:
    response_text = next(
        (cr["response"] for cr in chat_responses if cr["id"] == ck["chat_response_id"]),
        None,
    )
    if response_text:
        train_data.append({"keyword": ck["keyword"], "response": response_text})

print(f"✅ Datos cargados: {len(train_data)} registros")

# ================================
# 🔹 2️⃣ ENTRENAMIENTO NLP
# ================================
nlp = spacy.blank("es")

if "textcat" not in nlp.pipe_names:
    textcat = nlp.add_pipe("textcat", last=True)

for data in train_data:
    textcat.add_label(data["response"])

training_data = []
for data in train_data:
    training_data.append((data["keyword"], {"cats": {data["response"]: 1.0}}))

optimizer = nlp.begin_training()
for epoch in range(10):
    losses = {}
    for text, annotations in training_data:
        doc = nlp.make_doc(text)
        example = Example.from_dict(doc, annotations)
        nlp.update([example], drop=0.5, losses=losses)
    print(f"📌 Pérdidas en la época {epoch}: {losses}")

nlp.to_disk("modelo_chatbot")
print("✅ Modelo entrenado y guardado como 'modelo_chatbot'")

# ================================
# 🔹 3️⃣ FUNCIONES Y ESTADO
# ================================
nlp = spacy.load("modelo_chatbot")
user_states = {}


def extraer_telefono(texto):
    match = re.search(r"(\+51)?9\d{8}", texto)
    if match:
        numero = match.group(0)
        if not numero.startswith("+51"):
            numero = "+51" + numero
        return numero
    return None


# ================================
# 🔹 4️⃣ ENDPOINT FLASK
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

                if lat and long:
                    # Ubicación recibida correctamente
                    google_maps_url = f"https://www.google.com/maps?q={lat},{long}&hl=es-419&markers={lat},{long}"
                    res2 = requests.post(
                        "http://31.97.11.235:3001/lead",
                        json={
                            "phone": telefono,
                            "message": f"Se generó un ticket de soporte desde la ubicación: {google_maps_url}",
                        },
                    )

                    if res1.status_code == 200 and res2.status_code == 200:
                        user_states.pop(user_id, None)
                        return jsonify(
                            {"response": chat_responses[4]["response"], "status": True}
                        )
                    else:
                        user_states.pop(user_id, None)
                        return jsonify(
                            {
                                "response": "Hubo un problema al registrar tu ticket. Intenta más tarde.",
                                "status": False,
                            }
                        )
                else:
                    # No se recibió ubicación
                    res2 = requests.post(
                        "http://31.97.11.235:3001/lead",
                        json={
                            "phone": telefono,
                            "message": "Se generó un ticket de soporte, pero no se recibió información de ubicación.",
                        },
                    )
                    user_states.pop(user_id, None)
                    return jsonify(
                        {
                            "response": "Tu ticket fue registrado, pero no se recibió ubicación.",
                            "status": False,
                        }
                    )
            except Exception as e:
                user_states.pop(user_id, None)
                return jsonify(
                    {
                        "response": "Error conectando con el servidor de tickets.",
                        "status": False,
                    }
                )
        else:
            user_states.pop(user_id, None)
            return jsonify(
                {
                    "response": "No pude detectar tu número. Asegúrate de enviarlo en formato 9XXXXXXXX o +519XXXXXXXX. Iniciemos de nuevo.",
                    "status": False,
                }
            )

    # 2️⃣ NLP normal
    doc = nlp(mensaje)
    categorias = doc.cats
    mejor_respuesta = max(categorias, key=categorias.get)

    # Cambiar estado si corresponde
    if mejor_respuesta == chat_responses[3]["response"]:
        user_states[user_id] = "esperando_telefono"

    # Buscar el texto correcto desde la lista de respuestas
    respuesta_obj = next(
        (r for r in chat_responses if r["response"] == mejor_respuesta), None
    )
    respuesta_texto = (
        respuesta_obj["response"] if respuesta_obj else "Lo siento, no entendí eso."
    )

    return jsonify({"response": respuesta_texto, "status": False})


# ================================
# 🔹 5️⃣ CORRER LA APP
# ================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=6000, debug=True)
