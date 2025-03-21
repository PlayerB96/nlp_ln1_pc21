import spacy
from spacy.training.example import Example
from chatbot_ln1.infrastructure.database import Database
from chatbot_ln1.domain.chat import ChatKeyword

class ChatService:
    def __init__(self):
        self.db = Database.get_connection()
        self.cursor = self.db.cursor(dictionary=True)
        self.nlp = spacy.blank("es")

        if "textcat" not in self.nlp.pipe_names:
            self.textcat = self.nlp.add_pipe("textcat", last=True)

        self.load_data()
        self.train_model()

    def load_data(self):
        self.cursor.execute("""
            SELECT ck.keyword, cr.response , cr.type,  cr.content
            FROM chat_keywords ck
            JOIN chat_responses cr ON ck.chat_response_id = cr.id
        """)
        train_data = self.cursor.fetchall()
        print(train_data)
        for data in train_data:
            self.textcat.add_label(data["response"])
            # Si el tipo es 2, agrega el contenido
            if data["type"] == 2:
                data["content"] = data.get("content", "")
            self.train_data = train_data

    def train_model(self):
        training_data = [(data["keyword"], {"cats": {data["response"]: 1.0}}) for data in self.train_data]
        optimizer = self.nlp.begin_training()
        for epoch in range(20):
            losses = {}
            for text, annotations in training_data:
                doc = self.nlp.make_doc(text)
                example = Example.from_dict(doc, annotations)
                self.nlp.update([example], drop=0.5, losses=losses)
            print(f"📌 Pérdidas en la época {epoch}: {losses}")

        self.nlp.to_disk("modelo_chatbot")
        print("✅ Modelo entrenado y guardado como 'modelo_chatbot'")

    def predict(self, message):
        doc = self.nlp(message)
        categorias = doc.cats

        # Obtener la respuesta con mayor probabilidad
        best_response = max(categorias, key=categorias.get)

        # Buscar en los datos de entrenamiento la coincidencia con la respuesta
        for data in self.train_data:
            if data["response"] == best_response:
                content = data.get("content", "") if data["type"] == 2 else ""
                return {"message": best_response, "content": content}

        return {"message": "No entiendo", "content": ""}
