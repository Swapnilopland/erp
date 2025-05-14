# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import numpy as np
import pandas as pd
import sqlite3
from datetime import datetime
import requests
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
CORS(app)

# Configuration
ERP_API_BASE_URL = "https://your-erp-api.com/v1"
ERP_API_KEY = "your-api-key"

class ERPChatbot:
    def __init__(self):
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.knowledge_graph = self.load_knowledge_graph()
        self.faqs = self.load_faqs()
        self.init_db()

        self.erp_session = requests.Session()
        self.erp_session.headers.update({
            'Authorization': f'Bearer {ERP_API_KEY}',
            'Content-Type': 'application/json'
        })

    def load_knowledge_graph(self):
        with open('knowledge_graph.json') as f:
            data = json.load(f)
            for intent in data['intents']:
                intent['patterns_embeddings'] = self.sentence_model.encode(intent['patterns'])
            return data

    def load_faqs(self):
        return pd.read_csv('erp_faqs.csv')

    def init_db(self):
        conn = sqlite3.connect('erp_chatbot.db')
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS conversations
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      user_id TEXT,
                      message TEXT,
                      response TEXT,
                      intent TEXT,
                      subintent TEXT,
                      sentiment TEXT,
                      confidence REAL,
                      timestamp DATETIME)''')
        c.execute('''CREATE TABLE IF NOT EXISTS api_calls
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      conversation_id INTEGER,
                      endpoint TEXT,
                      parameters TEXT,
                      response_code INTEGER,
                      timestamp DATETIME)''')
        conn.commit()
        conn.close()

    def analyze_sentiment(self, text):
        lower = text.lower()
        if any(word in lower for word in ['good', 'great', 'excellent', 'happy']):
            return {'label': 'POSITIVE', 'score': 0.9}
        elif any(word in lower for word in ['bad', 'terrible', 'sad', 'angry']):
            return {'label': 'NEGATIVE', 'score': 0.9}
        return {'label': 'NEUTRAL', 'score': 0.6}

    def classify_intent(self, text):
        text_embedding = self.sentence_model.encode([text])
        best_intent = None
        highest_sim = 0
        for intent in self.knowledge_graph['intents']:
            similarities = cosine_similarity(text_embedding, intent['patterns_embeddings'])
            max_sim = np.max(similarities)
            if max_sim > highest_sim:
                highest_sim = max_sim
                best_intent = intent
        subintent = None
        if best_intent and 'subintents' in best_intent:
            for sub in best_intent['subintents']:
                sub_embeddings = self.sentence_model.encode(sub['patterns'])
                similarities = cosine_similarity(text_embedding, sub_embeddings)
                sub_sim = np.max(similarities)
                if sub_sim > 0.7:
                    subintent = sub['subintent']
                    highest_sim = min(highest_sim, sub_sim)
                    break
        return {
            'intent': best_intent['intent'] if best_intent else None,
            'subintent': subintent,
            'confidence': float(highest_sim)
        }

    def call_erp_api(self, endpoint, params=None, method='GET'):
        try:
            if method == 'POST':
                response = self.erp_session.post(
                    f"{ERP_API_BASE_URL}/{endpoint}",
                    json=params or {}
                )
            else:
                response = self.erp_session.get(
                    f"{ERP_API_BASE_URL}/{endpoint}",
                    params=params or {}
                )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"API Error: {e}")
            return None

    def generate_response(self, intent_data, user_message):
        intent = intent_data['intent']
        subintent = intent_data['subintent']
        matched_intent = next((i for i in self.knowledge_graph['intents'] if i['intent'] == intent), None)
        if not matched_intent:
            return self.get_fallback_response(user_message)

        if intent == "leave_balance":
            return "You have 8 casual leaves, 5 sick leaves, and 10 earned leaves remaining."
        if intent == "leave_history":
            return "Your recent leaves include:\n- 2024-12-05 to 2024-12-07 (Sick Leave)\n- 2025-01-10 to 2025-01-12 (Casual Leave)\n- 2025-03-15 (Earned Leave)"

        if intent == "leave_request":
            return self.apply_leave(user_message)

        if intent == "expenses":
            if subintent == "submit_expense":
                return "Sure, please provide the expense category and amount to proceed."
            elif subintent == "view_expense":
                return "Here is a list of your recent expenses:\n- Travel: $120\n- Meals: $45\n(For full report, visit your expense dashboard.)"
            elif subintent == "expense_status":
                return "Let me check the status of your recent expense claims. One moment please..."
            else:
                return "What would you like to do with your expenses? You can submit a new one or check existing records."

        if matched_intent.get('api_endpoint'):
            api_response = self.call_erp_api(
                matched_intent['api_endpoint'],
                self.extract_api_parameters(user_message, matched_intent)
            )
            if api_response:
                return self.format_api_response(api_response, matched_intent)

        if subintent and 'subintents' in matched_intent:
            matched_subintent = next((s for s in matched_intent['subintents'] if s['subintent'] == subintent), None)
            if matched_subintent:
                return matched_subintent['responses'][0]

        return matched_intent['responses'][0]

    def apply_leave(self, message):
        leave_data = {
            "leave_type": "Casual Leave",
            "start_date": "2025-06-01",
            "end_date": "2025-06-02",
            "reason": "Personal"
        }
        response = self.call_erp_api("apply_leave", leave_data, method='POST')
        if response:
            return "Your leave has been applied successfully."
        return "Failed to apply leave. Please try again later."

    def extract_api_parameters(self, text, intent):
        params = {}
        if 'parameters' in intent:
            for param in intent['parameters']:
                if param['type'] == 'date':
                    if 'today' in text.lower():
                        params[param['name']] = datetime.now().strftime('%Y-%m-%d')
                elif param['type'] == 'id':
                    words = text.split()
                    for word in words:
                        if word.isdigit():
                            params[param['name']] = word
                            break
        return params

    def format_api_response(self, api_data, intent):
        template = intent.get('response_template', '{data}')
        if intent['intent'] == 'inventory_query':
            items = api_data.get('items', [])
            if not items:
                return "No inventory items found matching your criteria."
            response = "Inventory levels:\n"
            for item in items[:5]:
                response += f"- {item['name']}: {item['quantity']} in stock\n"
            if len(items) > 5:
                response += f"\n({len(items) - 5} more items not shown)"
            return response
        return template.format(data=json.dumps(api_data, indent=2))

    def get_fallback_response(self, user_message):
        query_embedding = self.sentence_model.encode([user_message])
        faq_embeddings = self.sentence_model.encode(self.faqs['question'])
        similarities = cosine_similarity(query_embedding, faq_embeddings)
        max_idx = np.argmax(similarities)
        if similarities[0][max_idx] > 0.5:
            return self.faqs.iloc[max_idx]['answer']
        return "I'm not sure I understand. Could you please rephrase your question?"

    def log_conversation(self, user_id, message, response, intent_data, sentiment):
        conn = sqlite3.connect('erp_chatbot.db')
        c = conn.cursor()
        c.execute('''INSERT INTO conversations 
                     (user_id, message, response, intent, subintent, sentiment, confidence, timestamp)
                     VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
                  (user_id, message, response,
                   intent_data.get('intent'), intent_data.get('subintent'),
                   sentiment.get('label'), intent_data.get('confidence', 0),
                   datetime.now()))
        conn.commit()
        conn.close()

chatbot = ERPChatbot()

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    user_id = data.get('user_id', 'anonymous')
    message = data.get('message', '')
    sentiment = chatbot.analyze_sentiment(message)
    intent_data = chatbot.classify_intent(message)
    response = chatbot.generate_response(intent_data, message)
    chatbot.log_conversation(user_id, message, response, intent_data, sentiment)
    return jsonify({
        'response': response,
        'intent': intent_data['intent'],
        'subintent': intent_data.get('subintent'),
        'confidence': intent_data['confidence'],
        'sentiment': sentiment['label'],
        'sentiment_score': sentiment['score']
    })

@app.route('/api/faqs', methods=['GET'])
def get_faqs():
    faqs = chatbot.faqs.to_dict('records')
    return jsonify(faqs)

if __name__ == '__main__':
    app.run(debug=True)
