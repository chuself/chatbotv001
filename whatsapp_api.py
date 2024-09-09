import os
import requests

def send_whatsapp_message(to_phone_number, message):
    # Fetch token and phone number ID from environment variables
    TOKEN = os.getenv('WHATSAPP_API_TOKEN', 'EAAQ1DGiDChoBO5YhY5dJXHZCzPITgXBEvP0RcRtRWL9Y5fBjUkLfIwVwQzGFYVA1RuQNfuF7ewR6qaLGLOHNTeXwezLuSfOfhNe7oZCINHLSJXthwSMBfSdns1EfubNYzd09hZCJn7ZADIdUacvWXpkvm2wtx7WrkdR1ZAdrS25cdBwAzI11gYHZCtTrcbWJZAGXrxZAGq1SyH1b9kDqt9a6qrzzkgH0gKlWUqoZD')
    PHONE_NUMBER_ID = os.getenv('WHATSAPP_PHONE_NUMBER_ID', '255782284345')
    
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

# For testing purposes, you can replace the environment variables directly:
# send_whatsapp_message('1234567890', 'Hello, this is a test message!')
