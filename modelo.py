import mysql.connector
import spacy
from spacy.training.example import Example
from flask import Flask, request, jsonify

# Inicializar Flask
app = Flask(__name__)

# ==========================
# 🔹 1️⃣ CONECTAR A MYSQL 🔹
# ==========================
db_config = {
    "host": "172.16.0.134",  # Cambia según tu configuración
    "user": "developer",
    "password": "L4num3r01",
    "database": "extranet",
}

# Conectar a MySQL
conn = mysql.connector.connect(**db_config)
cursor = conn.cursor(dictionary=True)

# ==========================
# 🔹 2️⃣ OBTENER DATOS 🔹
# ==========================
cursor.execute(
    """
    SELECT ck.keyword, cr.response 
    FROM chat_keywords ck
    JOIN chat_responses cr ON ck.chat_response_id = cr.id
"""
)
train_data = cursor.fetchall()

# Cerrar conexión
cursor.close()
conn.close()

print(f"✅ Datos cargados: {len(train_data)} registros")

# ==========================
# 🔹 3️⃣ ENTRENAR MODELO NLP 🔹
# ==========================
# Cargar modelo base de español
nlp = spacy.blank("es")

# Agregar clasificador de texto
if "textcat" not in nlp.pipe_names:
    textcat = nlp.add_pipe("textcat", last=True)

# Agregar etiquetas dinámicas (respuestas únicas)
for data in train_data:
    textcat.add_label(data["response"])

# Preparar datos de entrenamiento
training_data = []
for data in train_data:
    training_data.append((data["keyword"], {"cats": {data["response"]: 1.0}}))

# Entrenar el modelo
optimizer = nlp.begin_training()
for epoch in range(10):
    losses = {}
    for text, annotations in training_data:
        doc = nlp.make_doc(text)
        example = Example.from_dict(doc, annotations)
        nlp.update([example], drop=0.5, losses=losses)
    print(f"📌 Pérdidas en la época {epoch}: {losses}")

# Guardar el modelo entrenado
nlp.to_disk("modelo_chatbot")
print("✅ Modelo entrenado y guardado como 'modelo_chatbot'")

# ==========================
# 🔹 4️⃣ DEFINIR ENDPOINT EN FLASK 🔹
# ==========================
# Cargar modelo entrenado
nlp = spacy.load("modelo_chatbot")


@app.route("/chatbot", methods=["POST"])
def chatbot():
    data = request.json
    mensaje = data.get("message", "")

    doc = nlp(mensaje)
    categorias = doc.cats
    mejor_respuesta = max(categorias, key=categorias.get)

    return jsonify({"response": mejor_respuesta})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=6000, debug=True)
