import os
import requests
import logging

# Set up logging to write to a file
log_file = 'whatsapp_api.log'  # You can change this to any file path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),  # Logs to a file
        logging.StreamHandler()  # Also logs to the console (optional)
    ]
)

def send_whatsapp_message(to_phone_number, message):
    # Fetch token and phone number ID from environment variables
    TOKEN = os.getenv('WHATSAPP_API_TOKEN')
    PHONE_NUMBER_ID = os.getenv('WHATSAPP_PHONE_NUMBER_ID')

    if not TOKEN or not PHONE_NUMBER_ID:
        logging.error("WhatsApp API token or phone number ID not set in environment variables")
        return

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
        response.raise_for_status()  # Raise an exception for 4XX/5XX status codes

        if response.status_code == 200:
            logging.info('Message sent successfully')
        else:
            logging.warning(f'Unexpected response code: {response.status_code}, Response: {response.text}')

    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to send message: {str(e)}")
        logging.error(f"Response: {response.text if response else 'No response received'}")

# Example usage for testing:
# send_whatsapp_message('1234567890', 'Hello, this is a test message!')

