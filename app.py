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




# import json
# import os
# import requests
# import logging
# import numpy as np
# import tensorflow as tf
# from flask import Flask, request, jsonify
# from nltk.stem import WordNetLemmatizer
# from sklearn.preprocessing import LabelEncoder
# import pickle
# import random

# import nltk
# from nltk.data import find
# from nltk.download import download

# def ensure_nltk_data():
#     try:
#         find('corpora/wordnet.zip')
#     except LookupError:
#         download('wordnet')

# ensure_nltk_data()



# app = Flask(__name__)

# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # Load the chatbot model and necessary files
# lemmatizer = WordNetLemmatizer()

# # Load intents JSON
# with open('intents.json') as file:
#     intents = json.load(file)

# # Load the trained model
# model = tf.keras.models.load_model('chatbotmodel.h5')

# # Load the tokenizer and label encoder
# with open('tokenizer.pickle', 'rb') as handle:
#     tokenizer = pickle.load(handle)

# with open('label_encoder.pickle', 'rb') as enc:
#     lbl_encoder = pickle.load(enc)

# # Define max sequence length (use the same value as during training)
# max_len = 20

# def preprocess_message(message):
#     """Preprocess the input message for prediction."""
#     message = lemmatizer.lemmatize(message.lower())
#     tokenized_message = tokenizer.texts_to_sequences([message])
#     padded_message = tf.keras.preprocessing.sequence.pad_sequences(tokenized_message, maxlen=max_len, padding='post')
#     return padded_message

# def get_response(predicted_class):
#     """Get the response based on the predicted class."""
#     intent = lbl_encoder.inverse_transform([predicted_class])[0]

#     # Find the corresponding response in intents.json
#     for i in intents['intents']:
#         if i['tag'] == intent:
#             return random.choice(i['responses'])
#     return "I am sorry, I do not understand that."

# def get_chatbot_response(user_message):
#     """Get response from the trained chatbot model."""
#     processed_message = preprocess_message(user_message)
#     prediction = model.predict(processed_message)
#     predicted_class = np.argmax(prediction)
#     return get_response(predicted_class)

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

#                                     # Get the chatbot's response based on the received message
#                                     response_message = get_chatbot_response(text_body)

#                                     # Send the chatbot's response back to the user
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


# import json
# import os
# import requests
# import logging
# import numpy as np
# import tensorflow as tf
# from flask import Flask, request, jsonify
# from nltk.stem import WordNetLemmatizer
# from sklearn.preprocessing import LabelEncoder
# import pickle
# import random
# import nltk

# # Ensure NLTK data is available
# def ensure_nltk_data():
#     try:
#         nltk.data.find('corpora/wordnet.zip')
#     except LookupError:
#         nltk.download('wordnet')
    
#     try:
#         nltk.data.find('corpora/omw-1.4.zip')
#     except LookupError:
#         nltk.download('omw-1.4')

# ensure_nltk_data()

# app = Flask(__name__)

# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# logger.info('WHATSAPP_API_TOKEN: %s', os.getenv('WHATSAPP_API_TOKEN'))
# logger.info('VERIFY_TOKEN: %s', os.getenv('VERIFY_TOKEN'))


# # Load the chatbot model and necessary files
# lemmatizer = WordNetLemmatizer()

# # Load intents JSON
# with open('intents.json') as file:
#     intents = json.load(file)

# # Load the trained model
# model = tf.keras.models.load_model('chatbotmodel.h5')

# # Load the tokenizer and label encoder
# with open('tokenizer.pickle', 'rb') as handle:
#     tokenizer = pickle.load(handle)
#     logger.info('Tokenizer loaded successfully')

# with open('label_encoder.pickle', 'rb') as enc:
#     lbl_encoder = pickle.load(enc)
#     logger.info('Label encoder loaded successfully')

# # Define max sequence length (use the same value as during training)
# max_len = 94  # Should match the model's expected input length

# def preprocess_message(message):
#     """Preprocess the input message for prediction."""
#     message = lemmatizer.lemmatize(message.lower())
#     tokenized_message = tokenizer.texts_to_sequences([message])
#     padded_message = tf.keras.preprocessing.sequence.pad_sequences(tokenized_message, maxlen=max_len, padding='post')
#     return padded_message

# def get_response(predicted_class):
#     """Get the response based on the predicted class."""
#     intent = lbl_encoder.inverse_transform([predicted_class])[0]

#     # Find the corresponding response in intents.json
#     for i in intents['intents']:
#         if i['tag'] == intent:
#             return random.choice(i['responses'])
#     return "I am sorry, I do not understand that."

# def get_chatbot_response(user_message):
#     """Get response from the trained chatbot model."""
#     processed_message = preprocess_message(user_message)
#     logger.info('Processed message: %s', processed_message)
#     prediction = model.predict(processed_message)
#     logger.info('Model prediction: %s', prediction)
#     predicted_class = np.argmax(prediction)
#     logger.info('Predicted class: %d', predicted_class)
#     return get_response(predicted_class)

# def send_whatsapp_message(to_phone_number, message):
#     TOKEN = os.getenv('WHATSAPP_API_TOKEN')
#     PHONE_NUMBER_ID = '426938520500591'

#     if not TOKEN:
#         logger.error("WHATSAPP_API_TOKEN not found in environment variables.")
#         return

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

#     logger.info('Sending message to %s with payload: %s', to_phone_number, payload)

#     try:
#         response = requests.post(url, headers=headers, json=payload)
#         response.raise_for_status()
#         logger.info('Message sent successfully')
#         logger.info('Response: %s', response.json())
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

# @app.route('/webhook', methods=['GET'])
# def webhook_get():
#     """Handle GET requests for webhook verification."""
#     verify_token = request.args.get('hub.verify_token')
#     challenge = request.args.get('hub.challenge')

#     logger.info('GET request received for webhook verification.')
#     logger.info('Verification token: %s', verify_token)
#     logger.info('Challenge: %s', challenge)

#     if verify_token == os.getenv('VERIFY_TOKEN'):
#         logger.info('Webhook verification successful.')
#         return challenge, 200
#     else:
#         logger.error('Webhook verification failed. Token does not match.')
#         return 'Forbidden', 403

# @app.route('/webhook', methods=['POST'])
# def webhook_post():
#     """Handle POST requests for receiving messages."""
#     try:
#         data = request.json
#         #logger.info('Received POST request with webhook data: %s', data)
#         logging.info(f"Received POST request with webhook data (trimmed): {str(data)[:500]}")

#         if 'entry' in data:
#             for entry in data['entry']:
#                 if 'changes' in entry:
#                     for change in entry['changes']:
#                         if 'value' in change and 'messages' in change['value']:
#                             messages = change['value']['messages']
#                             for message in messages:
#                                 from_number = message['from']
#                                 text_body = message['text']['body']
#                                 logger.info(f"Received message from {from_number}: {text_body}")

#                                 # Get the chatbot's response based on the received message
#                                 response_message = get_chatbot_response(text_body)
#                                 logger.info(f"Response to send: {response_message}")

#                                 # Send the chatbot's response back to the user
#                                 send_whatsapp_message(from_number, response_message)
                                

#         return jsonify({'status': 'received'}), 200

#     except KeyError as e:
#         logger.error(f'Missing key in webhook data: {e}')
#         return jsonify({'status': 'error', 'message': 'Missing key in webhook data'}), 400
#     except Exception as e:
#         logger.error(f'Error processing webhook: {e}')
#         return jsonify({'status': 'error', 'message': str(e)}), 500

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000)













# import json
# import os
# import requests
# import logging
# import numpy as np
# import tensorflow as tf
# from flask import Flask, request, jsonify
# from nltk.stem import WordNetLemmatizer
# from sklearn.preprocessing import LabelEncoder
# import pickle
# import random
# import nltk

# # Ensure NLTK data is available
# def ensure_nltk_data():
#     try:
#         nltk.data.find('corpora/wordnet.zip')
#     except LookupError:
#         nltk.download('wordnet')
    
#     try:
#         nltk.data.find('corpora/omw-1.4.zip')
#     except LookupError:
#         nltk.download('omw-1.4')

# ensure_nltk_data()

# app = Flask(__name__)

# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# logger.info('WHATSAPP_API_TOKEN: %s', os.getenv('WHATSAPP_API_TOKEN'))
# logger.info('VERIFY_TOKEN: %s', os.getenv('VERIFY_TOKEN'))

# # Load the chatbot model and necessary files
# lemmatizer = WordNetLemmatizer()

# # Load intents JSON
# with open('intents.json') as file:
#     intents = json.load(file)

# # Load the trained model
# model = tf.keras.models.load_model('chatbotmodel.h5')

# # Load the tokenizer and label encoder
# with open('tokenizer.pickle', 'rb') as handle:
#     tokenizer = pickle.load(handle)
#     logger.info('Tokenizer loaded successfully')

# with open('label_encoder.pickle', 'rb') as enc:
#     lbl_encoder = pickle.load(enc)
#     logger.info('Label encoder loaded successfully')

# # Define max sequence length (use the same value as during training)
# max_len = 94  # Should match the model's expected input length

# def preprocess_message(message):
#     """Preprocess the input message for prediction."""
#     message = lemmatizer.lemmatize(message.lower())
#     tokenized_message = tokenizer.texts_to_sequences([message])
#     padded_message = tf.keras.preprocessing.sequence.pad_sequences(tokenized_message, maxlen=max_len, padding='post')
#     return padded_message

# def get_response(predicted_class):
#     """Get the response based on the predicted class."""
#     intent = lbl_encoder.inverse_transform([predicted_class])[0]

#     # Find the corresponding response in intents.json
#     for i in intents['intents']:
#         if i['tag'] == intent:
#             return random.choice(i['responses'])
#     return "I am sorry, I do not understand that."

# def get_chatbot_response(user_message):
#     """Get response from the trained chatbot model."""
#     processed_message = preprocess_message(user_message)
#     logger.info('Processed input for model: %s', processed_message)
#     prediction = model.predict(processed_message)
#     logger.info('Model prediction: %s', prediction)
#     predicted_class = np.argmax(prediction)
#     logger.info('Predicted class: %d', predicted_class)
#     return get_response(predicted_class)

# def send_whatsapp_message(to_phone_number, message):
#     TOKEN = os.getenv('WHATSAPP_API_TOKEN')
#     PHONE_NUMBER_ID = '426938520500591'

#     if not TOKEN:
#         logger.error("WHATSAPP_API_TOKEN not found in environment variables.")
#         return

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

#     logger.info('Sending message to %s with payload: %s', to_phone_number, payload)

#     try:
#         response = requests.post(url, headers=headers, json=payload)
#         if response.status_code == 200:
#             logger.info(f"Successfully sent message to {to_phone_number}")
#         else:
#             logger.error(f"Failed to send message. Status code: {response.status_code}, Response: {response.text}")
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

# @app.route('/webhook', methods=['GET'])
# def webhook_get():
#     """Handle GET requests for webhook verification."""
#     verify_token = request.args.get('hub.verify_token')
#     challenge = request.args.get('hub.challenge')

#     logger.info('GET request received for webhook verification.')
#     logger.info('Verification token: %s', verify_token)
#     logger.info('Challenge: %s', challenge)

#     if verify_token == os.getenv('VERIFY_TOKEN'):
#         logger.info('Webhook verification successful.')
#         return challenge, 200
#     else:
#         logger.error('Webhook verification failed. Token does not match.')
#         return 'Forbidden', 403

# @app.route('/webhook', methods=['POST'])
# def webhook_post():
#     """Handle POST requests for receiving messages."""
#     try:
#         data = request.json
#         logging.info(f"Received POST request with webhook data (trimmed): {str(data)[:500]}")

#         if 'entry' in data:
#             for entry in data['entry']:
#                 if 'changes' in entry:
#                     for change in entry['changes']:
#                         if 'value' in change and 'messages' in change['value']:
#                             messages = change['value']['messages']
#                             for message in messages:
#                                 from_number = message['from']
#                                 text_body = message['text']['body']
#                                 logger.info(f"Received message from {from_number}: {text_body}")

#                                 # Get the chatbot's response based on the received message
#                                 response_message = get_chatbot_response(text_body)
#                                 logger.info(f"Response to send: {response_message}")

#                                 # Send the chatbot's response back to the user
#                                 send_whatsapp_message(from_number, response_message)
                                

#         return jsonify({'status': 'received'}), 200

#     except KeyError as e:
#         logger.error(f'Missing key in webhook data: {e}')
#         logger.error(f'Webhook data: {data}')  # Log the full data for better debugging
#         return jsonify({'status': 'error', 'message': 'Missing key in webhook data'}), 400
#     except Exception as e:
#         logger.error(f'Error processing webhook: {e}')
#         logger.error(f'Webhook data: {data}')  # Log the full data for better debugging
#         return jsonify({'status': 'error', 'message': str(e)}), 500

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000)












# import os
# from flask import Flask, request, jsonify
# from ChatBot1 import predict_class, get_response, log_interaction, fuzzy_match, match_key_phrases, intents
# import logging
# from datetime import datetime

# # Initialize Flask app
# app = Flask(__name__)

# # Set up logging for troubleshooting purposes
# log_file_name = f"flask_chatbot_{datetime.now().strftime('%Y-%m-%d')}.log"
# logging.basicConfig(filename=log_file_name, level=logging.DEBUG, 
#                     format='%(asctime)s - %(levelname)s - %(message)s')

# # Logging startup
# logging.info("Flask app started. Waiting for webhook calls...")

# # Endpoint for WhatsApp Webhook (for POST requests)
# @app.route('/webhook', methods=['POST'])
# def webhook():
#     try:
#         # Log incoming request
#         logging.info("Received a POST request from WhatsApp")

#         # Extract the incoming message from the POST request
#         data = request.get_json()
#         logging.debug(f"Request data: {data}")

#         if 'message' not in data:
#             logging.error("No 'message' field in POST data")
#             return jsonify({"error": "No message field found"}), 400

#         user_input = data['message'].strip()

#         # Handle empty messages
#         if not user_input:
#             logging.info("Received an empty message")
#             return jsonify({"response": "Please enter something!"}), 200

#         logging.info(f"Processing user input: {user_input}")

#         # First, run fuzzy matching to catch possible mistyped key phrases or patterns
#         patterns = []
#         for intent in intents['intents']:
#             patterns.extend(intent['patterns'])

#         fuzzy_matched_pattern = fuzzy_match(user_input, patterns)

#         if fuzzy_matched_pattern:
#             message = fuzzy_matched_pattern  # Use fuzzy matched pattern
#             logging.info(f"Fuzzy matched pattern: {message}")
#             ints, raw_predictions = predict_class(message)
#         else:
#             # If no fuzzy match, check key phrases
#             matched_department = match_key_phrases(user_input)

#             if matched_department:
#                 logging.info(f"Matched department via key phrase: {matched_department}")
#                 response = get_response([{'intent': matched_department, 'probability': '1.0'}], intents)
#                 log_interaction(user_input, response, predicted_intents=[{'intent': matched_department, 'probability': '1.0'}], model_predictions=None)
#                 return jsonify({"response": response}), 200

#             # If no key phrase or fuzzy match, proceed with regular intent prediction
#             logging.info("No key phrase or fuzzy match found. Proceeding with intent prediction.")
#             ints, raw_predictions = predict_class(user_input)

#         # Get response based on predicted intents
#         if ints[0]['intent'] == 'fallback':
#             response = "Sorry, I don't understand. Could you rephrase that?"
#             logging.warning(f"Fallback triggered for input: {user_input}")
#         else:
#             response = get_response(ints, intents)
#             logging.info(f"Generated response: {response}")

#         # Log the interaction
#         log_interaction(user_input, response, predicted_intents=ints, model_predictions=raw_predictions)

#         # Return the response as a JSON object
#         return jsonify({"response": response}), 200

#     except Exception as e:
#         # Log the error for debugging purposes
#         logging.error(f"Error processing webhook: {str(e)}", exc_info=True)
#         return jsonify({"error": str(e)}), 500

# # Health check route to ensure the server is running
# @app.route('/health', methods=['GET'])
# def health():
#     return jsonify({"status": "Healthy"}), 200

# # Run the Flask app
# if __name__ == '__main__':
#     # Print to console to know that Flask has started
#     print("Flask server is running...")
#     logging.info("Flask server is up and running.")
#     app.run(debug=True)



import os
import json
import logging
from flask import Flask, request, jsonify
from ChatBot1 import predict_class, get_response, log_interaction, fuzzy_match, match_key_phrases, intents
from datetime import datetime
import pickle
import sys

# Path to the script
nltk_script = os.path.join(os.path.dirname(__file__), 'download_nltk_data.py')
os.system(f'{sys.executable} {nltk_script}')


# Initialize Flask app
app = Flask(__name__)

# Set up logging for troubleshooting purposes
log_file_name = f"flask_chatbot_{datetime.now().strftime('%Y-%m-%d')}.log"
logging.basicConfig(filename=log_file_name, level=logging.DEBUG, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Logging startup
logging.info("Flask app started. Waiting for webhook calls...")

# Load configuration
with open('config.json', 'r') as config_file:
    config = json.load(config_file)

# Load model and other resources
def load_resources():
    with open(config['words_file'], 'rb') as f:
        words = pickle.load(f)
    with open(config['classes_file'], 'rb') as f:
        classes = pickle.load(f)
    return words, classes

words, classes = load_resources()

# Endpoint for WhatsApp Webhook (for POST requests)
@app.route('/webhook', methods=['POST'])
def webhook():
    try:
        # Log incoming request
        logging.info("Received a POST request from WhatsApp")

        # Extract the incoming message from the POST request
        data = request.get_json()
        logging.debug(f"Request data: {data}")

        if 'entry' not in data or not data['entry']:
            logging.error("No 'entry' field in POST data")
            return jsonify({"error": "Invalid data format"}), 400

        # Extract the message body and sender
        message_body = data['entry'][0]['changes'][0]['value']['messages'][0]['text']['body']
        sender = data['entry'][0]['changes'][0]['value']['messages'][0]['from']
        
        logging.info(f"Message from {sender}: {message_body}")

        # Handle empty messages
        if not message_body.strip():
            logging.info("Received an empty message")
            return jsonify({"response": "Please enter something!"}), 200

        logging.info(f"Processing user input: {message_body}")

        # First, run fuzzy matching to catch possible mistyped key phrases or patterns
        patterns = []
        for intent in intents['intents']:
            patterns.extend(intent['patterns'])

        fuzzy_matched_pattern = fuzzy_match(message_body, patterns)

        if fuzzy_matched_pattern:
            message = fuzzy_matched_pattern  # Use fuzzy matched pattern
            logging.info(f"Fuzzy matched pattern: {message}")
            ints, raw_predictions = predict_class(message)
        else:
            # If no fuzzy match, check key phrases
            matched_department = match_key_phrases(message_body)

            if matched_department:
                logging.info(f"Matched department via key phrase: {matched_department}")
                response = get_response([{'intent': matched_department, 'probability': '1.0'}], intents)
                log_interaction(message_body, response, predicted_intents=[{'intent': matched_department, 'probability': '1.0'}], model_predictions=None)
                return jsonify({"response": response}), 200

            # If no key phrase or fuzzy match, proceed with regular intent prediction
            logging.info("No key phrase or fuzzy match found. Proceeding with intent prediction.")
            ints, raw_predictions = predict_class(message_body)

        # Get response based on predicted intents
        if ints[0]['intent'] == 'fallback':
            response = "Sorry, I don't understand. Could you rephrase that?"
            logging.warning(f"Fallback triggered for input: {message_body}")
        else:
            response = get_response(ints, intents)
            logging.info(f"Generated response: {response}")

        # Log the interaction
        log_interaction(message_body, response, predicted_intents=ints, model_predictions=raw_predictions)

        # Return the response as a JSON object
        return jsonify({"response": response}), 200

    except Exception as e:
        # Log the error for debugging purposes
        logging.error(f"Error processing webhook: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

# Health check route to ensure the server is running
@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "Healthy"}), 200

# Run the Flask app
if __name__ == '__main__':
    # Print to console to know that Flask has started
    print("Flask server is running...")
    logging.info("Flask server is up and running.")
    app.run(debug=True, host='0.0.0.0', port=5000)

