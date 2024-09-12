# from flask import Flask, request, jsonify
# import os
# import requests

# app = Flask(__name__)

# def send_whatsapp_message(to_phone_number, message):
#     TOKEN = os.getenv('WHATSAPP_API_TOKEN')
#     PHONE_NUMBER_ID = os.getenv('WHATSAPP_PHONE_NUMBER_ID')
#     url = f'https://graph.facebook.com/v13.0/{PHONE_NUMBER_ID}/messages'
#     headers = {
#         'Authorization': f'Bearer {TOKEN}',
#         'Content-Type': 'application/json',
#     }
#     payload = {
#         'messaging_product': 'whatsapp',
#         'to': to_phone_number,
#         'text': {'body': message},
#     }
#     response = requests.post(url, headers=headers, json=payload)
#     if response.status_code == 200:
#         print('Message sent successfully')
#     else:
#         print(f'Failed to send message: {response.text}')

# @app.route('/')
# def home():
#     return "Welcome to the WhatsApp Chatbot made By Chuself!"


# @app.route('/webhook', methods=['POST'])
# def webhook():
#     data = request.json
#     # Process the incoming message
#     print(data)
#     # Here you would process the incoming message and respond using your chatbot logic
#     # For now, just send a test message as an example
#     send_whatsapp_message('1234567890', 'Hello, this is a test message!')
#     return jsonify({'status': 'received'}), 200

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000)




# from flask import Flask, request, jsonify
# import os
# import requests

# app = Flask(__name__)

# def send_whatsapp_message(to_phone_number, message):
#     TOKEN = os.getenv('WHATSAPP_API_TOKEN')
#     PHONE_NUMBER_ID = os.getenv('WHATSAPP_PHONE_NUMBER_ID')
#     url = f'https://graph.facebook.com/v13.0/{PHONE_NUMBER_ID}/messages'
#     headers = {
#         'Authorization': f'Bearer {TOKEN}',
#         'Content-Type': 'application/json',
#     }
#     payload = {
#         'messaging_product': 'whatsapp',
#         'to': to_phone_number,
#         'text': {'body': message},
#     }
#     response = requests.post(url, headers=headers, json=payload)
#     if response.status_code == 200:
#         print('Message sent successfully')
#     else:
#         print(f'Failed to send message: {response.text}')

# from flask import Flask, request, jsonify
# import os
# import requests
# import logging

# app = Flask(__name__)

# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# def send_whatsapp_message(to_phone_number, message):
#     TOKEN = os.getenv('WHATSAPP_API_TOKEN')
#     PHONE_NUMBER_ID = '426938520500591'

#     if not TOKEN:
#         logger.error("WHATSAPP_API_TOKEN not found in environment variables.")
#         return

#     # Ensure the phone number is in the international format
#     if not to_phone_number.startswith('2'):
#         to_phone_number = f"255{to_phone_number[1:]}"
    
#     url = f'https://graph.facebook.com/v13.0/{PHONE_NUMBER_ID}/messages'
#     headers = {
#         'Authorization': f'Bearer {TOKEN}',
#         'Content-Type': 'application/json',
#     }
#     payload = {
#         'messaging_product': 'whatsapp',
#         'to': to_phone_number,
#         'text': {'body': message},
#     }
    
#     try:
#         response = requests.post(url, headers=headers, json=payload)
#         response.raise_for_status()  # Raises HTTPError for bad responses
#         logger.info('Message sent successfully')
#     except requests.exceptions.HTTPError as http_err:
#         logger.error(f'HTTP error occurred: {http_err}')
#         logger.error(f'Response content: {response.text}')
#     except requests.exceptions.RequestException as req_err:
#         logger.error(f'Request error occurred: {req_err}')
#     except Exception as err:
#         logger.error(f'Other error occurred: {err}')

# @app.route('/')
# def home():
#     return "Welcome to the WhatsApp Chatbot!"

# @app.route('/webhook', methods=['GET', 'POST'])
# def webhook():
#     if request.method == 'GET':
#         # Verification step for Facebook webhook
#         verify_token = request.args.get('hub.verify_token')
#         challenge = request.args.get('hub.challenge')
#         if verify_token == os.getenv('VERIFY_TOKEN'):
#             return challenge, 200
#         else:
#             return 'Forbidden', 403

#     if request.method == 'POST':
#         try:
#             data = request.json
#             logger.info('Received webhook data: %s', data)

#             if 'entry' in data:
#                 for entry in data['entry']:
#                     if 'changes' in entry:
#                         for change in entry['changes']:
#                             if 'value' in change and 'messages' in change['value']:
#                                 messages = change['value']['messages']
#                                 for message in messages:
#                                     from_number = message['from']
#                                     text_body = message['text']['body']
#                                     logger.info(f"Received message from {from_number}: {text_body}")

#                                     # Process the incoming message and send a response
#                                     response_message = f"Received your message: {text_body}"# Parrots the text i sent back to me
#                                     send_whatsapp_message(from_number, response_message)

#             return jsonify({'status': 'received'}), 200

#         except KeyError as e:
#             logger.error(f'Missing key in webhook data: {e}')
#             return jsonify({'status': 'error', 'message': 'Missing key in webhook data'}), 400
#         except Exception as e:
#             logger.error(f'Error processing webhook: {e}')
#             return jsonify({'status': 'error', 'message': str(e)}), 500

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000)

from flask import Flask, request, jsonify
import os
import requests
import logging
import json
import numpy as np
import tensorflow as tf
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
import random

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the chatbot model and necessary files
lemmatizer = WordNetLemmatizer()

# Load intents JSON
with open('intents.json') as file:
    intents = json.load(file)

# Load the trained model
model = tf.keras.models.load_model('chatbot_model.h5')

# Load the tokenizer and label encoder
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

with open('label_encoder.pickle', 'rb') as enc:
    lbl_encoder = pickle.load(enc)

# Define max sequence length (use the same value as during training)
max_len = 20

def preprocess_message(message):
    """Preprocess the input message for prediction."""
    message = lemmatizer.lemmatize(message.lower())
    tokenized_message = tokenizer.texts_to_sequences([message])
    padded_message = tf.keras.preprocessing.sequence.pad_sequences(tokenized_message, maxlen=max_len, padding='post')
    return padded_message

def get_chatbot_response(user_message):
    """Get response from the trained chatbot model."""
    processed_message = preprocess_message(user_message)
    prediction = model.predict(processed_message)
    predicted_class = np.argmax(prediction)
    intent = lbl_encoder.inverse_transform([predicted_class])[0]

    # Find the corresponding response in intents.json
    for i in intents['intents']:
        if i['tag'] == intent:
            return random.choice(i['responses'])
    return "I am sorry, I do not understand that."

def send_whatsapp_message(to_phone_number, message):
    TOKEN = os.getenv('WHATSAPP_API_TOKEN')
    PHONE_NUMBER_ID = '426938520500591'

    if not TOKEN:
        logger.error("WHATSAPP_API_TOKEN not found in environment variables.")
        return

    # Ensure the phone number is in the international format
    if not to_phone_number.startswith('2'):
        to_phone_number = f"255{to_phone_number[1:]}"
    
    url = f'https://graph.facebook.com/v13.0/{PHONE_NUMBER_ID}/messages'
    headers = {
        'Authorization': f'Bearer {TOKEN}',
        'Content-Type': 'application/json',
    }
    payload = {
        'messaging_product': 'whatsapp',
        'to': to_phone_number,
        'text': {'body': message},
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()  # Raises HTTPError for bad responses
        logger.info('Message sent successfully')
    except requests.exceptions.HTTPError as http_err:
        logger.error(f'HTTP error occurred: {http_err}')
        logger.error(f'Response content: {response.text}')
    except requests.exceptions.RequestException as req_err:
        logger.error(f'Request error occurred: {req_err}')
    except Exception as err:
        logger.error(f'Other error occurred: {err}')

@app.route('/')
def home():
    return "Welcome to the WhatsApp Chatbot!"

@app.route('/webhook', methods=['GET', 'POST'])
def webhook():
    if request.method == 'GET':
        # Verification step for Facebook webhook
        verify_token = request.args.get('hub.verify_token')
        challenge = request.args.get('hub.challenge')
        if verify_token == os.getenv('VERIFY_TOKEN'):
            return challenge, 200
        else:
            return 'Forbidden', 403

    if request.method == 'POST':
        try:
            data = request.json
            logger.info('Received webhook data: %s', data)

            if 'entry' in data:
                for entry in data['entry']:
                    if 'changes' in entry:
                        for change in entry['changes']:
                            if 'value' in change and 'messages' in change['value']:
                                messages = change['value']['messages']
                                for message in messages:
                                    from_number = message['from']
                                    text_body = message['text']['body']
                                    logger.info(f"Received message from {from_number}: {text_body}")

                                    # Get the chatbot's response
                                    response_message = get_chatbot_response(text_body)
                                    send_whatsapp_message(from_number, response_message)

            return jsonify({'status': 'received'}), 200

        except KeyError as e:
            logger.error(f'Missing key in webhook data: {e}')
            return jsonify({'status': 'error', 'message': 'Missing key in webhook data'}), 400
        except Exception as e:
            logger.error(f'Error processing webhook: {e}')
            return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

