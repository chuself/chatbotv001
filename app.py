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
from flask import Flask, request, jsonify
import os
import requests

app = Flask(__name__)

def send_whatsapp_message(to_phone_number, message):
    TOKEN = os.getenv('WHATSAPP_API_TOKEN')
    PHONE_NUMBER_ID = os.getenv('WHATSAPP_PHONE_NUMBER_ID')
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
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code == 200:
        print('Message sent successfully')
    else:
        print(f'Failed to send message: {response.text}')

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
        # Process the incoming message
        print(data)
        # Here you would process the incoming message and respond using your chatbot logic
        # For now, just send a test message as an example
        send_whatsapp_message('1234567890', 'Hello, this is a test message!')
        return jsonify({'status': 'received'}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

