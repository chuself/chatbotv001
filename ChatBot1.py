# import os
# print(os.getcwd())
# os.chdir('d:\Programing\Projects\ChatBotV1')

# import os
# import random
# import json
# import pickle
# import numpy as np
# import nltk
# from nltk.stem import WordNetLemmatizer
# from tensorflow.keras.models import load_model
# import difflib  # For fuzzy matching

# lemmatizer = WordNetLemmatizer()

# try:
#     with open('intents.json') as f:
#         intents = json.load(f)
#     print("Intents loaded successfully.")
    
#     with open('words.pkl', 'rb') as f:
#         words = pickle.load(f)
#     print("Words loaded successfully.")
    
#     with open('classes.pkl', 'rb') as f:
#         classes = pickle.load(f)
#     print("Classes loaded successfully.")
    
#     model = load_model('chatbotmodel.h5')
#     print("Model loaded successfully.")
    
# except Exception as e:
#     print(f"Error loading files or model: {str(e)}")
#     exit()


# # Initialize context management
# context = {}

# # Fuzzy matching function
# def fuzzy_match(user_input, known_words):
#     patterns = []
#     for intent in intents['intents']:
#         patterns.extend(intent['patterns'])
    
#     # Perform fuzzy matching
#     matched_word = difflib.get_close_matches(user_input, known_words, n=1, cutoff=0.7)
#     print(f"Fuzzy Match Result: {matched_word}")  # Print the results of fuzzy matching
#     return matched_word


# def clean_up_sentence(sentence):
#     sentence_words = nltk.word_tokenize(sentence)
#     sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
#     return sentence_words

# def bag_of_words(sentence):
#     sentence_words = clean_up_sentence(sentence)
#     bag = [0] * len(words)

#     # For debugging purposes
#     print(f"Total number of words in the model: {len(words)}")  
#     print(f"Sentence words: {sentence_words}")

#     for w in sentence_words:
#         for i, word in enumerate(words):
#             if word == w:
#                 bag[i] = 1
#     return np.array(bag)

# def predict_class(sentence):
#     bow = bag_of_words(sentence)
#     res = model.predict(np.array([bow]))[0]
#     ERROR_THRESHOLD = 0.75
#     results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

#     if not results:
#         return [{'intent': 'fallback', 'probability': '1.0'}]  # Return fallback if no matches

#     results.sort(key=lambda x: x[1], reverse=True)
#     return [{'intent': classes[r[0]], 'probability': str(r[1])} for r in results]

# def get_response(intents_list, intents_json):
#     tag = intents_list[0]['intent']

#     list_of_intents = intents_json['intents']
#     for i in list_of_intents:
#         if i['tag'] == tag:
#             result = random.choice(i['responses'])
#             break
    
#     # Update context if needed
#     if tag == 'joke':
#         context['context'] = 'joke_context'
#     else:
#         context['context'] = None

#     # Add response variation here (change based on context, time, etc.)
#     if context['context'] == 'joke_context':
#         result = "Another joke? Here's a good one: " + result

#     return result

# # Validation of input
# def is_input_valid(sentence_words):
#     return any(word in words for word in sentence_words)

# print("Nexus is Online")

# while True:
#     message = input("")

#     if not message.strip():  # Skip if message is empty or just spaces
#         print("Please enter something!")
#         continue

#     # First, check for fuzzy matches with known words
#     fuzzy_matched_word = fuzzy_match(message, words)
#     if fuzzy_matched_word:
#         message = fuzzy_matched_word[0]  # Update the message with the closest match
#     else:
#         # If no fuzzy match is found, notify the user or handle it
#         print("Sorry, I couldn't understand that.")
#         continue
   
#     sentence_words = clean_up_sentence(message)

#     # Validate the input after fuzzy matching
#     if not is_input_valid(sentence_words):
#         print("Sorry, I couldn't understand that.")
#         continue

#     ints = predict_class(message)

#     # If the top intent is fallback, provide a default response
#     if ints[0]['intent'] == 'fallback':
#         print("Sorry, I don't understand. Could you rephrase that?")
#         continue

#     res = get_response(ints, intents)

#     print(res)
    
#     # Reduced debugging info (optional: you can uncomment these for full debugging)
#     print(f"Predicted intents: {ints}")
#     # Print context to debug
#     print(f"Current context: {context}")
#     print(f"Current context: {context}")
#     print(f"Model prediction: {model.predict(np.array([bag_of_words(message)]))}")

# import os
# import random
# import json
# import pickle
# import numpy as np
# import nltk
# from nltk.stem import WordNetLemmatizer
# from tensorflow.keras.models import load_model
# import difflib  # For fuzzy matching
# from datetime import datetime

# # Load configuration from config file
# with open('config.json') as f:
#     config = json.load(f)

# lemmatizer = WordNetLemmatizer()

# # Function to dynamically generate log file name based on the current date
# def get_log_file_path():
#     current_date = datetime.now().strftime("%Y-%m-%d")
#     log_file_name = f"chatbot_log_{current_date}.txt"
#     return log_file_name

# # Function to log interaction to a file, including additional model details
# def log_interaction(user_input, bot_response, predicted_intents=None, model_predictions=None):
#     log_file_path = get_log_file_path()  # Generate file name based on the current date
    
#     with open(log_file_path, "a") as log_file:
#         timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
#         # Log user input and bot response
#         log_file.write(f"[{timestamp}] User: {user_input}\n")
#         log_file.write(f"[{timestamp}] Bot: {bot_response}\n")
        
#         # Log predicted intents and model raw predictions (if available)
#         if predicted_intents:
#             log_file.write(f"[{timestamp}] Predicted Intents: {predicted_intents}\n")
#         if model_predictions is not None:
#             log_file.write(f"[{timestamp}] Model Predictions: {model_predictions}\n")
        
#         log_file.write("\n")  # Empty line to separate interactions

# try:
#     # Load intents
#     with open(config['intents_file']) as f:
#         intents = json.load(f)
#     print("Intents loaded successfully.")
    
#     # Load words
#     with open(config['words_file'], 'rb') as f:
#         words = pickle.load(f)
#     print("Words loaded successfully.")
    
#     # Load classes
#     with open(config['classes_file'], 'rb') as f:
#         classes = pickle.load(f)
#     print("Classes loaded successfully.")
    
#     # Load model
#     model = load_model(config['model_file'])
#     print("Model loaded successfully.")
    
#     # Print model summary for debugging
#     print("\nModel Summary:")
#     model.summary()
    
# except Exception as e:
#     print(f"Error loading files or model: {str(e)}")
#     exit()

# # Initialize context management
# context = {}

# # Fuzzy matching function
# def fuzzy_match(user_input, patterns):
#     matched_pattern = difflib.get_close_matches(user_input, patterns, n=1, cutoff=config['fuzzy_match_cutoff'])
#     print(f"Fuzzy Match Result: {matched_pattern}")  # Print the results of fuzzy matching
#     return matched_pattern[0] if matched_pattern else None

# def clean_up_sentence(sentence):
#     sentence_words = nltk.word_tokenize(sentence)
#     sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
#     return sentence_words

# def bag_of_words(sentence):
#     sentence_words = clean_up_sentence(sentence)
#     bag = [0] * len(words)

#     # For debugging purposes
#     print(f"Total number of words in the model: {len(words)}")  
#     print(f"Sentence words: {sentence_words}")

#     for w in sentence_words:
#         for i, word in enumerate(words):
#             if word == w:
#                 bag[i] = 1
#     return np.array(bag)

# def predict_class(sentence):
#     bow = bag_of_words(sentence)
#     res = model.predict(np.array([bow]))[0]
#     #print(f"Model raw predictions: {res}")  # Debugging: print raw model predictions
    
#     ERROR_THRESHOLD = config['error_threshold']  # Use threshold from config
#     results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

#     if not results:
#         return [{'intent': 'fallback', 'probability': '1.0'}]  # Return fallback if no matches

#     results.sort(key=lambda x: x[1], reverse=True)
#     intents = [{'intent': classes[r[0]], 'probability': str(r[1])} for r in results]
    
#     # Debugging: Print the classes and intent mappings
#     #print(f"Classes: {classes}")
#     print(f"Predicted intents: {intents}")
    
#     return intents , res

# def get_response(intents_list, intents_json):
#     tag = intents_list[0]['intent']
#     if len(intents_list) > 1 and float(intents_list[0]['probability']) - float(intents_list[1]['probability']) < 0.1:
#         return "I am not quite sure. Did you mean to say 'bye' or 'hello'?"

#     # Find the corresponding response from the intents.json file
#     for i in intents_json['intents']:
#         if i['tag'] == tag:
#             return random.choice(i['responses'])
    
#     # In case something goes wrong
#     return "Sorry, I did not understand that. Could you rephrase?"

#     # Update context if needed
#     if tag == 'joke':
#         context['context'] = 'joke_context'
#     else:
#         context['context'] = None

#     # Add response variation here (change based on context, time, etc.)
#     if context['context'] == 'joke_context':
#         result = "Another joke? Here's a good one: " + result

#     return result

# # Validation of input
# def is_input_valid(sentence_words):
#     return any(word in words for word in sentence_words)

# print("Nexus is Online")

# while True:
#     original_message = input("User:   ")  # Capture original user input

#     if not original_message.strip():  # Skip if message is empty or just spaces
#         print("Nexus: Please enter something!")
#         continue

#     # # First, check for fuzzy matches with known words
#     # fuzzy_matched_word = fuzzy_match(original_message, words)
#     # if fuzzy_matched_word:
#     #     message = " ".join(fuzzy_matched_word)#[0]  # Reconstruct the sentence from matched words
#     #     print(f"Reconstructed sentence from fuzzy matches: {message}")
#     # else:
#     #     # If no fuzzy match is found, notify the user or handle it
#     #     print("Nexus: Sorry, I couldn't understand that.")
#     #     log_interaction(original_message, res, predicted_intents=ints, model_predictions=raw_predictions)
#     #     continue

#     # Extract patterns from intents
#     patterns = []
#     for intent in intents['intents']:
#         patterns.extend(intent['patterns'])

#     # First, check for fuzzy matches with patterns
#     fuzzy_matched_pattern = fuzzy_match(original_message, patterns)
#     if fuzzy_matched_pattern:
#         message = fuzzy_matched_pattern  # Use the fuzzy matched pattern directly
#         print(f"Using fuzzy matched pattern: {message}")
#     else:
#         # If no fuzzy match is found, notify the user or handle it
#         print("Nexus: Sorry, I couldn't understand that.")
#         log_interaction(original_message, "Nexus: Sorry, I couldn't understand that.", predicted_intents=None, model_predictions=None)
#         continue

#     ints ,raw_predictions = predict_class(message)

#     # If the top intent is fallback, provide a default response
#     if ints[0]['intent'] == 'fallback':
#         print("Nexus: Sorry, I don't understand. Could you rephrase that?")
#         continue

#     res = get_response(ints, intents)
#     print("Nexus: ", res)

#     log_interaction(original_message, res, predicted_intents=ints, model_predictions=raw_predictions)
    
#     # Reduced debugging info (optional: you can uncomment these for full debugging)
#     print(f"Predicted intents: {ints}")
#     print(f"Current context: {context}")
#     print(f"Model prediction: {model.predict(np.array([bag_of_words(message)]))}")


 






import os
import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
import difflib  # For fuzzy matching
from datetime import datetime

# Load configuration from config file, including key phrases
with open('config.json') as f:
    config = json.load(f)

# Load key phrases from the config
key_phrases = config['key_phrases']  # New: Key phrases for each department

lemmatizer = WordNetLemmatizer()

# Function to dynamically generate log file name based on the current date
def get_log_file_path():
    current_date = datetime.now().strftime("%Y-%m-%d")
    log_file_name = f"chatbot_log_{current_date}.txt"
    return log_file_name

# Function to log interaction to a file, including additional model details
def log_interaction(user_input, bot_response, predicted_intents=None, model_predictions=None):
    log_file_path = get_log_file_path()  # Generate file name based on the current date
    
    with open(log_file_path, "a") as log_file:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Log user input and bot response
        log_file.write(f"[{timestamp}] User: {user_input}\n")
        log_file.write(f"[{timestamp}] Bot: {bot_response}\n")
        
        # Log predicted intents and model raw predictions (if available)
        if predicted_intents:
            log_file.write(f"[{timestamp}] Predicted Intents: {predicted_intents}\n")
        if model_predictions is not None:
            log_file.write(f"[{timestamp}] Model Predictions: {model_predictions}\n")
        
        log_file.write("\n")  # Empty line to separate interactions

try:
    # Load intents
    with open(config['intents_file']) as f:
        intents = json.load(f)
    print("Intents loaded successfully.")
    
    # Load words
    with open(config['words_file'], 'rb') as f:
        words = pickle.load(f)
    print("Words loaded successfully.")
    
    # Load classes
    with open(config['classes_file'], 'rb') as f:
        classes = pickle.load(f)
    print("Classes loaded successfully.")
    
    # Load model
    model = load_model(config['model_file'])
    print("Model loaded successfully.")
    
    # Print model summary for debugging
    print("\nModel Summary:")
    model.summary()
    
except Exception as e:
    print(f"Error loading files or model: {str(e)}")
    exit()

# Initialize context management
context = {}

# Fuzzy matching function
def fuzzy_match(user_input, patterns):
    matched_pattern = difflib.get_close_matches(user_input, patterns, n=1, cutoff=config['fuzzy_match_cutoff'])
    print(f"Fuzzy Match Result: {matched_pattern}")  # Print the results of fuzzy matching
    return matched_pattern[0] if matched_pattern else None

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)

    # For debugging purposes
    print(f"Total number of words in the model: {len(words)}")  
    print(f"Sentence words: {sentence_words}")

    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    
    ERROR_THRESHOLD = config['error_threshold']  # Use threshold from config
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    if not results:
        return [{'intent': 'fallback', 'probability': '1.0'}]  # Return fallback if no matches

    results.sort(key=lambda x: x[1], reverse=True)
    intents = [{'intent': classes[r[0]], 'probability': str(r[1])} for r in results]
    
    # Debugging: Print the predicted intents
    print(f"Predicted intents: {intents}")
    
    return intents, res

def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    if len(intents_list) > 1 and float(intents_list[0]['probability']) - float(intents_list[1]['probability']) < 0.1:
        return "I am not quite sure. Did you mean to say 'bye' or 'hello'?"

    # Find the corresponding response from the intents.json file
    for i in intents_json['intents']:
        if i['tag'] == tag:
            return random.choice(i['responses'])
    
    # In case something goes wrong
    return "Sorry, I did not understand that. Could you rephrase?"

    # Update context if needed
    if tag == 'joke':
        context['context'] = 'joke_context'
    else:
        context['context'] = None

    # Add response variation here (change based on context, time, etc.)
    if context['context'] == 'joke_context':
        result = "Another joke? Here's a good one: " + result

    return result

# Function to match user input against key phrases
def match_key_phrases(user_input):
    # Loop through each department's key phrases to find a match
    for department, phrases in key_phrases.items():
        for phrase in phrases:
            if phrase.lower() in user_input.lower():  # Check if the key phrase is present in user input
                print(f"Matched key phrase '{phrase}' for department '{department}'")  # Debugging info
                return department  # Return the department (intent) if a match is found
    return None  # Return None if no key phrase match

print("Nexus is Online")

while True:
    original_message = input("User:   ")  # Capture original user input

    if not original_message.strip():  # Skip if message is empty or just spaces
        print("Nexus: Please enter something!")
        continue

    # First, run fuzzy matching to catch possible mistyped key phrases or patterns
    patterns = []
    for intent in intents['intents']:
        patterns.extend(intent['patterns'])

    fuzzy_matched_pattern = fuzzy_match(original_message, patterns)

    if fuzzy_matched_pattern:
        message = fuzzy_matched_pattern  # Use the fuzzy matched pattern directly
        print(f"Using fuzzy matched pattern: {message}")
        ints, raw_predictions = predict_class(message)
    else:
        # If no fuzzy match, check key phrases
        matched_department = match_key_phrases(original_message)

        if matched_department:
            print(f"Nexus: Identified department based on key phrase match: {matched_department}")
            response = get_response([{'intent': matched_department, 'probability': '1.0'}], intents)
            print("Nexus:", response)
            log_interaction(original_message, response, predicted_intents=[{'intent': matched_department, 'probability': '1.0'}], model_predictions=None)
            continue

        # If no key phrase or fuzzy match, proceed with regular intent prediction
        ints, raw_predictions = predict_class(original_message)

    # Get response based on predicted intents
    if ints[0]['intent'] == 'fallback':
        print("Nexus: Sorry, I don't understand. Could you rephrase that?")
        continue

    res = get_response(ints, intents)
    print("Nexus:", res)
    log_interaction(original_message, res, predicted_intents=ints, model_predictions=raw_predictions)
