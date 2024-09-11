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

from flask import Flask, request, jsonify
import os
import requests
import logging

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def send_whatsapp_message(to_phone_number, message):
    TOKEN = os.getenv('WHATSAPP_API_TOKEN')  # Make sure this is your permanent token
    PHONE_NUMBER_ID = '439621585894029'  # Your actual Phone Number ID
    
    # Ensure the phone number is in the international format
    if not to_phone_number.startswith('+'):
        to_phone_number = f"+255{to_phone_number[1:]}"
    
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
        response.raise_for_status()  # Raise an exception for HTTP errors
        logging.info('Message sent successfully')
        return response.json()
    except requests.exceptions.HTTPError as http_err:
        logging.error(f'HTTP error occurred: {http_err}')
    except requests.exceptions.ConnectionError as conn_err:
        logging.error(f'Connection error occurred: {conn_err}')
    except requests.exceptions.Timeout as timeout_err:
        logging.error(f'Timeout error occurred: {timeout_err}')
    except requests.exceptions.RequestException as req_err:
        logging.error(f'Request exception occurred: {req_err}')
    except Exception as e:
        logging.error(f'An unexpected error occurred: {e}')

def generate_response(user_message):
    """
    Generate a response based on the user's message.
    """
    user_message = user_message.lower()
    if 'hello' in user_message:
        return 'Hi there! How can I assist you today?'
    elif 'help' in user_message:
        return 'Sure, I am here to help. What do you need assistance with?'
    elif 'bye' in user_message:
        return 'Goodbye! Have a great day!'
    else:
        return 'I am not sure how to respond to that. Can you please provide more details?'

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
        data = request.json
        logging.info('Received webhook data: %s', data)
        
        # Extract message and sender details
        if 'messages' in data['value']:
            messages = data['value']['messages']
            for message in messages:
                from_number = message['from']
                message_body = message['text']['body']
                
                # Generate a response based on the received message
                response_message = generate_response(message_body)
                
                # Send the response message back to the user
                send_whatsapp_message(from_number, response_message)
        
        return jsonify({'status': 'received'}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
