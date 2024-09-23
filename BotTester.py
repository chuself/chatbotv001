# import json
# import difflib
# import datetime
# import ChatBot1  # Import the chatbot script

# # Import necessary functions from ChatBot1
# predict_class = ChatBot1.predict_class
# get_response = ChatBot1.get_response

# # Load intents
# with open('intents.json') as file:
#     intents = json.load(file)

# def get_log_file_path():
#     """Generates the log file path based on the current date."""
#     current_date = datetime.datetime.now().strftime("%Y-%m-%d")
#     log_file_name = f"chatbottester_log_{current_date}.txt"
#     return log_file_name

# def log_interaction(pattern, fuzzy_match, predicted_intents, response, model_predictions):
#     """Logs the chatbot interaction to a file."""
#     try:
#         log_file_path = get_log_file_path()
#         print(f"Logging interaction to: {log_file_path}")  # Debug print
#         with open(log_file_path, 'a') as log_file:  # Open the log file in append mode
#             timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#             log_file.write(f"[{timestamp}] Pattern: {pattern}\n")
#             log_file.write(f"[{timestamp}] Fuzzy Match: {fuzzy_match}\n")
#             log_file.write(f"[{timestamp}] Predicted Intents: {predicted_intents}\n")
#             log_file.write(f"[{timestamp}] Response: {response}\n")
#             log_file.write(f"[{timestamp}] Model Predictions: {model_predictions}\n")
#             log_file.write('-' * 40 + '\n')
#         print(f"Interaction logged for pattern: {pattern}")  # Debug print
#     except Exception as e:
#         print(f"Error logging interaction: {e}")

# def get_fuzzy_match(user_input, patterns):
#     """Returns the closest fuzzy match for a user input from the patterns."""
#     matched_pattern = difflib.get_close_matches(user_input, patterns, n=1, cutoff=0.8)
#     return matched_pattern[0] if matched_pattern else None

# def run_bot_tester():
#     """Runs the test for each pattern in intents.json and logs the results."""
#     print("Bot tester is running...")  # Debug print
#     for intent in intents['intents']:
#         for pattern in intent['patterns']:
#             print(f"Testing pattern: {pattern}")  # Debug print
#             # Perform fuzzy matching for the pattern
#             fuzzy_match = get_fuzzy_match(pattern, [p for intent in intents['intents'] for p in intent['patterns']])

#             # Predict the intent and get the model's response
#             predicted_intents, model_predictions = predict_class(pattern)
#             response = get_response(predicted_intents, intents)

#             # Log the interaction
#             log_interaction(pattern, fuzzy_match, predicted_intents, response, model_predictions)
    
#     print("Bot tester finished.")  # Debug print

# if __name__ == "__main__":
#     # Check if we can execute the tester before the chatbot starts
#     try:
#         # Step 1: Run the bot tester and log interactions
#         print("Starting bot tester...")  # Debug print
#         run_bot_tester()

#         # Step 2: After tester finishes, start the chatbot loop
#         print("\nBot tester completed. Starting the chatbot...\n")

#         # Only call the chatbot main loop after tester completes
#         ChatBot1.main()

#     except Exception as e:
#         print(f"Error running BotTester: {e}")




# import random
# import json
# import pickle
# import numpy as np
# import nltk
# from nltk.stem import WordNetLemmatizer
# from tensorflow.keras.models import load_model
# import difflib
# from datetime import datetime

# # Load configuration
# with open('config.json') as f:
#     config = json.load(f)

# lemmatizer = WordNetLemmatizer()

# # Load resources based on configuration
# with open(config['intents_file']) as f:
#     intents = json.load(f)
# print("Intents loaded successfully.")

# with open(config['words_file'], 'rb') as f:
#     words = pickle.load(f)
# print("Words loaded successfully.")

# with open(config['classes_file'], 'rb') as f:
#     classes = pickle.load(f)
# print("Classes loaded successfully.")

# model = load_model(config['model_file'])
# print("Model loaded successfully.")

# def fuzzy_match(user_input, patterns):
#     matched_pattern = difflib.get_close_matches(user_input, patterns, n=1, cutoff=config['fuzzy_match_cutoff'])
#     return matched_pattern[0] if matched_pattern else None

# def clean_up_sentence(sentence):
#     sentence_words = nltk.word_tokenize(sentence)
#     sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
#     return sentence_words

# def bag_of_words(sentence):
#     sentence_words = clean_up_sentence(sentence)
#     bag = [0] * len(words)
#     for w in sentence_words:
#         for i, word in enumerate(words):
#             if word == w:
#                 bag[i] = 1
#     return np.array(bag)

# def predict_class(sentence):
#     bow = bag_of_words(sentence)
#     res = model.predict(np.array([bow]))[0]
#     results = [[i, r] for i, r in enumerate(res) if r > config['error_threshold']]
#     if not results:
#         return [{'intent': 'fallback', 'probability': '1.0'}], res
#     results.sort(key=lambda x: x[1], reverse=True)
#     intents = [{'intent': classes[r[0]], 'probability': str(r[1])} for r in results]
#     return intents, res

# def get_response(intents_list, intents_json):
#     tag = intents_list[0]['intent']
#     if len(intents_list) > 1 and float(intents_list[0]['probability']) - float(intents_list[1]['probability']) < 0.1:
#         return "I am not quite sure. Did you mean to say 'bye' or 'hello'?"
#     for i in intents_json['intents']:
#         if i['tag'] == tag:
#             return random.choice(i['responses'])
#     return "Sorry, I did not understand that. Could you rephrase?"

# def get_log_file_path():
#     current_date = datetime.now().strftime("%Y-%m-%d")
#     log_file_name = f"Bot_tester_log_{current_date}.txt"
#     return log_file_name

# def log_interaction(input_text, predicted_intents, response):
#     log_file_path = get_log_file_path()
#     with open(log_file_path, "a") as log_file:
#         timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#         log_file.write(f"[{timestamp}] Input: {input_text}\n")
#         log_file.write(f"[{timestamp}] Predicted Intents: {predicted_intents}\n")
#         log_file.write(f"[{timestamp}] Response: {response}\n")
#         log_file.write("\n")

# # Iterate over each pattern in the intents file and test
# for intent in intents['intents']:
#     for pattern in intent['patterns']:
#         print(f"Testing pattern: {pattern}")
#         fuzzy_matched_pattern = fuzzy_match(pattern, [p for i in intents['intents'] for p in i['patterns']])
#         if fuzzy_matched_pattern:
#             message = fuzzy_matched_pattern
#         else:
#             message = pattern  # Use the original pattern if no fuzzy match is found

#         ints, raw_predictions = predict_class(message)
        
#         if ints[0]['intent'] == 'fallback':
#             response = "Sorry, I don't understand. Could you rephrase that?"
#         else:
#             response = get_response(ints, intents)

#         print(f"Pattern: {pattern}")
#         print(f"Response: {response}")
        
#         # Log the interaction
#         log_interaction(pattern, ints, response)






# import random
# import json
# import pickle
# import numpy as np
# import nltk
# from nltk.stem import WordNetLemmatizer
# from tensorflow.keras.models import load_model
# import difflib
# from datetime import datetime

# # Load configuration
# with open('config.json') as f:
#     config = json.load(f)

# lemmatizer = WordNetLemmatizer()

# # Load resources based on configuration
# with open(config['intents_file']) as f:
#     intents = json.load(f)
# print("Intents loaded successfully.")

# with open(config['words_file'], 'rb') as f:
#     words = pickle.load(f)
# print("Words loaded successfully.")

# with open(config['classes_file'], 'rb') as f:
#     classes = pickle.load(f)
# print("Classes loaded successfully.")

# model = load_model(config['model_file'])
# print("Model loaded successfully.")

# def fuzzy_match(user_input, patterns):
#     matched_pattern = difflib.get_close_matches(user_input, patterns, n=1, cutoff=config['fuzzy_match_cutoff'])
#     return matched_pattern[0] if matched_pattern else None

# def clean_up_sentence(sentence):
#     sentence_words = nltk.word_tokenize(sentence)
#     sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
#     return sentence_words

# def bag_of_words(sentence):
#     sentence_words = clean_up_sentence(sentence)
#     bag = [0] * len(words)
#     for w in sentence_words:
#         for i, word in enumerate(words):
#             if word == w:
#                 bag[i] = 1
#     return np.array(bag)

# def predict_class(sentence):
#     bow = bag_of_words(sentence)
#     res = model.predict(np.array([bow]))[0]
#     results = [[i, r] for i, r in enumerate(res) if r > config['error_threshold']]
#     if not results:
#         return [{'intent': 'fallback', 'probability': '1.0'}], res
#     results.sort(key=lambda x: x[1], reverse=True)
#     intents = [{'intent': classes[r[0]], 'probability': str(r[1])} for r in results]
#     return intents, res

# def get_response(intents_list, intents_json):
#     tag = intents_list[0]['intent']
#     if len(intents_list) > 1 and float(intents_list[0]['probability']) - float(intents_list[1]['probability']) < 0.1:
#         return tag, "I am not quite sure. Did you mean to say 'bye' or 'hello'?", tag
#     for i in intents_json['intents']:
#         if i['tag'] == tag:
#             return tag, random.choice(i['responses']), i['tag']
#     return tag, "Sorry, I did not understand that. Could you rephrase?", None

# def get_log_file_path():
#     current_date = datetime.now().strftime("%Y-%m-%d")
#     log_file_name = f"Bot_tester_log_{current_date}.txt"
#     return log_file_name

# def log_interaction(input_text, predicted_intents, response, input_tag, response_tag, tags_match):
#     log_file_path = get_log_file_path()
#     with open(log_file_path, "a") as log_file:
#         timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#         log_file.write(f"[{timestamp}] Input: {input_text}\n")
#         log_file.write(f"[{timestamp}] Predicted Intents: {predicted_intents}\n")
#         log_file.write(f"[{timestamp}] Response: {response}\n")
#         log_file.write(f"[{timestamp}] Input Tag: {input_tag}\n")
#         log_file.write(f"[{timestamp}] Response Tag: {response_tag}\n")
#         if not tags_match:
#             log_file.write(f"[{timestamp}] Error: The tags don't match\n")
#         log_file.write("\n")

# # Iterate over each pattern in the intents file and test
# for intent in intents['intents']:
#     for pattern in intent['patterns']:
#         print(f"Testing pattern: {pattern}")
#         fuzzy_matched_pattern = fuzzy_match(pattern, [p for i in intents['intents'] for p in i['patterns']])
#         if fuzzy_matched_pattern:
#             message = fuzzy_matched_pattern
#         else:
#             message = pattern  # Use the original pattern if no fuzzy match is found

#         ints, raw_predictions = predict_class(message)
        
#         if ints[0]['intent'] == 'fallback':
#             response = "Sorry, I don't understand. Could you rephrase that?"
#             input_tag = ints[0]['intent']
#             response_tag = None
#             tags_match = False
#         else:
#             input_tag, response, response_tag = get_response(ints, intents)
#             tags_match = (input_tag == response_tag)

#         print(f"Pattern: {pattern}")
#         print(f"Response: {response}")
        
#         # Log the interaction
#         log_interaction(pattern, ints, response, input_tag, response_tag, tags_match)



# import random
# import json
# import pickle
# import numpy as np
# import nltk
# from nltk.stem import WordNetLemmatizer
# from tensorflow.keras.models import load_model
# import difflib
# from datetime import datetime
# import string

# # Load configuration
# with open('config.json') as f:
#     config = json.load(f)

# lemmatizer = WordNetLemmatizer()

# # Load resources based on configuration
# with open(config['intents_file']) as f:
#     intents = json.load(f)
# print("Intents loaded successfully.")

# with open(config['words_file'], 'rb') as f:
#     words = pickle.load(f)
# print("Words loaded successfully.")

# with open(config['classes_file'], 'rb') as f:
#     classes = pickle.load(f)
# print("Classes loaded successfully.")

# model = load_model(config['model_file'])
# print("Model loaded successfully.")

# def fuzzy_match(user_input, patterns):
#     matched_pattern = difflib.get_close_matches(user_input, patterns, n=1, cutoff=config['fuzzy_match_cutoff'])
#     return matched_pattern[0] if matched_pattern else None

# def clean_up_sentence(sentence):
#     sentence_words = nltk.word_tokenize(sentence)
#     sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
#     return sentence_words

# def bag_of_words(sentence):
#     sentence_words = clean_up_sentence(sentence)
#     bag = [0] * len(words)
#     for w in sentence_words:
#         for i, word in enumerate(words):
#             if word == w:
#                 bag[i] = 1
#     return np.array(bag)

# def predict_class(sentence):
#     bow = bag_of_words(sentence)
#     res = model.predict(np.array([bow]))[0]
#     results = [[i, r] for i, r in enumerate(res) if r > config['error_threshold']]
#     if not results:
#         return [{'intent': 'fallback', 'probability': '1.0'}], res
#     results.sort(key=lambda x: x[1], reverse=True)
#     intents = [{'intent': classes[r[0]], 'probability': str(r[1])} for r in results]
#     return intents, res

# def get_response(intents_list, intents_json):
#     tag = intents_list[0]['intent']
#     if len(intents_list) > 1 and float(intents_list[0]['probability']) - float(intents_list[1]['probability']) < 0.1:
#         return "I am not quite sure. Did you mean to say 'bye' or 'hello'?"
#     for i in intents_json['intents']:
#         if i['tag'] == tag:
#             return random.choice(i['responses'])
#     return "Sorry, I did not understand that. Could you rephrase?"

# def get_log_file_path():
#     current_date = datetime.now().strftime("%Y-%m-%d")
#     log_file_name = f"Bot_tester_log_{current_date}.txt"
#     return log_file_name

# def log_incorrect_interaction(input_text, variations, predicted_intents, response, input_tag, response_tag):
#     log_file_path = get_log_file_path()
#     with open(log_file_path, "a") as log_file:
#         timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#         log_file.write(f"[{timestamp}] Pattern: {input_text}\n")
#         log_file.write(f"[{timestamp}] Variations: {', '.join(variations)}\n")
#         log_file.write(f"[{timestamp}] Predicted Intents: {predicted_intents}\n")
#         log_file.write(f"[{timestamp}] Response: {response}\n")
#         log_file.write(f"[{timestamp}] Input Tag: {input_tag}\n")
#         log_file.write(f"[{timestamp}] Response Tag: {response_tag}\n")
#         log_file.write("\n")

# # Generate variations of a pattern
# def generate_variations(pattern):
#     variations = [pattern]
#     words = pattern.split()
#     # Reorder words
#     variations.append(' '.join(reversed(words)))
#     # Simulate typos
#     for i in range(len(words)):
#         typo_variations = words[:]
#         typo_variations[i] = ''.join(c if c not in string.ascii_lowercase else random.choice(string.ascii_lowercase) for c in typo_variations[i])
#         variations.append(' '.join(typo_variations))
#     return variations

# # Start session logging
# log_file_path = get_log_file_path()
# with open(log_file_path, "w") as log_file:
#     log_file.write(f"Session started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

# # Initialize counters
# correct_variations = 0
# incorrect_variations = 0

# # Iterate over each pattern in the intents file and test
# for intent in intents['intents']:
#     for pattern in intent['patterns']:
#         variations = generate_variations(pattern)
#         print(f"Testing pattern: {pattern}")
#         all_variations_correct = True

#         for variation in variations:
#             fuzzy_matched_pattern = fuzzy_match(variation, [p for i in intents['intents'] for p in i['patterns']])
#             if fuzzy_matched_pattern:
#                 message = fuzzy_matched_pattern
#             else:
#                 message = variation  # Use the variation if no fuzzy match is found

#             ints, raw_predictions = predict_class(message)
            
#             if ints[0]['intent'] == 'fallback':
#                 response = "Sorry, I don't understand. Could you rephrase that?"
#                 response_tag = 'fallback'
#                 all_variations_correct = False
#             else:
#                 response = get_response(ints, intents)
#                 response_tag = ints[0]['intent']

#             input_tag = intent['tag']

#             # Check if the response tag matches the input tag
#             if response_tag == input_tag:
#                 correct_variations += 1
#             else:
#                 incorrect_variations += 1
#                 log_incorrect_interaction(pattern, variations, ints, response, input_tag, response_tag)
#                 all_variations_correct = False

# # End session logging with counters
# with open(log_file_path, "a") as log_file:
#     log_file.write(f"Session ended at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
#     log_file.write(f"Correct Variations: {correct_variations}\n")
#     log_file.write(f"Incorrect Variations: {incorrect_variations}\n")








# import random
# import json
# import pickle
# import numpy as np
# import nltk
# from nltk.stem import WordNetLemmatizer
# from tensorflow.keras.models import load_model
# import difflib
# from datetime import datetime
# import string

# # Load configuration
# with open('config.json') as f:
#     config = json.load(f)

# lemmatizer = WordNetLemmatizer()

# # Load resources
# with open(config['intents_file']) as f:
#     intents = json.load(f)

# with open(config['words_file'], 'rb') as f:
#     words = pickle.load(f)

# with open(config['classes_file'], 'rb') as f:
#     classes = pickle.load(f)

# model = load_model(config['model_file'])

# def get_log_file_path():
#     current_date = datetime.now().strftime("%Y-%m-%d")
#     log_file_name = f"Bot_tester_log_{current_date}.txt"
#     return log_file_name

# def clean_up_sentence(sentence):
#     sentence_words = nltk.word_tokenize(sentence)
#     sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
#     return sentence_words

# def bag_of_words(sentence):
#     sentence_words = clean_up_sentence(sentence)
#     bag = [0] * len(words)
#     for w in sentence_words:
#         for i, word in enumerate(words):
#             if word == w:
#                 bag[i] = 1
#     return np.array(bag)

# def detect_department(sentence):
#     for key_phrase, department in config['key_phrases'].items():
#         if key_phrase.lower() in sentence.lower():
#             return department
#     return "Unknown"

# def predict_class(sentence):
#     bow = bag_of_words(sentence)
#     res = model.predict(np.array([bow]))[0]
#     results = [[i, r] for i, r in enumerate(res) if r > config['error_threshold']]
#     if not results:
#         return [{'intent': 'fallback', 'probability': '1.0'}], res, "Unknown"
#     results.sort(key=lambda x: x[1], reverse=True)
#     intents = [{'intent': classes[r[0]], 'probability': str(r[1])} for r in results]
    
#     # Detect department
#     detected_department = detect_department(sentence)
#     return intents, res, detected_department

# def get_response(intents_list, intents_json):
#     tag = intents_list[0]['intent']
#     for i in intents_json['intents']:
#         if i['tag'] == tag:
#             return random.choice(i['responses'])
#     return "Sorry, I did not understand that."

# def log_incorrect_detection(input_text, variations, predicted_intents, response, input_tag, response_tag, detected_department):
#     log_file_path = get_log_file_path()
#     with open(log_file_path, "a") as log_file:
#         timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#         log_file.write(f"[{timestamp}] Pattern: {input_text}\n")
#         log_file.write(f"[{timestamp}] Variations: {', '.join(variations)}\n")
#         log_file.write(f"[{timestamp}] Predicted Intents: {predicted_intents}\n")
#         log_file.write(f"[{timestamp}] Response: {response}\n")
#         log_file.write(f"[{timestamp}] Input Tag: {input_tag}\n")
#         log_file.write(f"[{timestamp}] Response Tag: {response_tag}\n")
#         log_file.write(f"[{timestamp}] Detected Department: {detected_department}\n")
#         log_file.write("\n")

# # Generate variations of a pattern
# def generate_variations(pattern):
#     variations = [pattern]
#     words = pattern.split()
#     # Reorder words
#     variations.append(' '.join(reversed(words)))
#     # Simulate typos
#     for i in range(len(words)):
#         typo_variations = words[:]
#         typo_variations[i] = ''.join(c if c not in string.ascii_lowercase else random.choice(string.ascii_lowercase) for c in typo_variations[i])
#         variations.append(' '.join(typo_variations))
#     return variations

# # Start session logging
# log_file_path = get_log_file_path()
# with open(log_file_path, "w") as log_file:
#     log_file.write(f"Session started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

# # Initialize counters
# correct_variations = 0
# incorrect_variations = 0

# # Iterate over each pattern in the intents file and test
# for intent in intents['intents']:
#     for pattern in intent['patterns']:
#         variations = generate_variations(pattern)
#         print(f"Testing pattern: {pattern}")
#         all_variations_correct = True

#         for variation in variations:
#             message = variation

#             ints, raw_predictions, detected_department = predict_class(message)
#             response = get_response(ints, intents)
#             input_tag = intent['tag']
#             response_tag = ints[0]['intent'] if ints else 'fallback'

#             # Check if the response tag matches the input tag
#             if response_tag == input_tag:
#                 correct_variations += 1
#             else:
#                 incorrect_variations += 1
#                 log_incorrect_detection(pattern, variations, ints, response, input_tag, response_tag, detected_department)
#                 all_variations_correct = False

# # End session logging with counters
# with open(log_file_path, "a") as log_file:
#     log_file.write(f"Session ended at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
#     log_file.write(f"Correct Variations: {correct_variations}\n")
#     log_file.write(f"Incorrect Variations: {incorrect_variations}\n")


import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
from datetime import datetime
import string

# Load configuration
with open('config.json') as f:
    config = json.load(f)

lemmatizer = WordNetLemmatizer()

# Load resources
with open(config['intents_file']) as f:
    intents = json.load(f)

with open(config['words_file'], 'rb') as f:
    words = pickle.load(f)

with open(config['classes_file'], 'rb') as f:
    classes = pickle.load(f)

model = load_model(config['model_file'])

def get_log_file_path():
    current_date = datetime.now().strftime("%Y-%m-%d")
    log_file_name = f"Bot_tester_log_{current_date}.txt"
    return log_file_name

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def detect_department(sentence):
    recognized_phrases = {}
    for key_phrase, department in config['key_phrases'].items():
        if key_phrase.lower() in sentence.lower():
            recognized_phrases[key_phrase] = department
            
    return recognized_phrases if recognized_phrases else {"Unknown": "Unknown"}

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    results = [[i, r] for i, r in enumerate(res) if r > config['error_threshold']]
    
    if not results:
        return [{'intent': 'fallback', 'probability': '1.0'}], res, {"Unknown": "Unknown"}
    
    results.sort(key=lambda x: x[1], reverse=True)
    intents = [{'intent': classes[r[0]], 'probability': str(r[1])} for r in results]
    
    detected_departments = detect_department(sentence)
    return intents, res, detected_departments

def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    for i in intents_json['intents']:
        if i['tag'] == tag:
            return random.choice(i['responses'])
    return "Sorry, I did not understand that."

def log_incorrect_detection(input_text, variations, predicted_intents, response, input_tag, response_tag, detected_departments):
    log_file_path = get_log_file_path()
    with open(log_file_path, "a") as log_file:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_file.write(f"[{timestamp}] Pattern: {input_text}\n")
        log_file.write(f"[{timestamp}] Variations: {', '.join(variations)}\n")
        log_file.write(f"[{timestamp}] Predicted Intents: {predicted_intents}\n")
        log_file.write(f"[{timestamp}] Response: {response}\n")
        log_file.write(f"[{timestamp}] Input Tag: {input_tag}\n")
        log_file.write(f"[{timestamp}] Response Tag: {response_tag}\n")
        log_file.write(f"[{timestamp}] Detected Departments: {detected_departments}\n")
        log_file.write("\n")

def log_key_phrase_detection(input_text, recognized_phrases):
    log_file_path = get_log_file_path()
    with open(log_file_path, "a") as log_file:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for key_phrase, department in recognized_phrases.items():
            log_file.write(f"[{timestamp}] Recognized Key Phrase: '{key_phrase}' in Department: {department} for input: '{input_text}'\n")
        log_file.write("\n")

def generate_variations(pattern):
    variations = [pattern]
    words = pattern.split()
    variations.append(' '.join(reversed(words)))  # Reorder words
    
    # Simulate typos
    for i in range(len(words)):
        typo_variations = words[:]
        typo_variations[i] = ''.join(c if c not in string.ascii_lowercase else random.choice(string.ascii_lowercase) for c in typo_variations[i])
        variations.append(' '.join(typo_variations))
    
    return variations

# Start session logging
log_file_path = get_log_file_path()
with open(log_file_path, "w") as log_file:
    log_file.write(f"Session started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

# Initialize counters
correct_variations = 0
incorrect_variations = 0

# Iterate over each pattern in the intents file and test
for intent in intents['intents']:
    for pattern in intent['patterns']:
        variations = generate_variations(pattern)
        print(f"Testing pattern: {pattern}")
        all_variations_correct = True

        for variation in variations:
            message = variation

            ints, raw_predictions, detected_departments = predict_class(message)
            response = get_response(ints, intents)
            input_tag = intent['tag']
            response_tag = ints[0]['intent'] if ints else 'fallback'

            # Log key phrase detections
            log_key_phrase_detection(pattern, detected_departments)

            # Check if the response tag matches the input tag
            if response_tag == input_tag:
                correct_variations += 1
            else:
                incorrect_variations += 1
                log_incorrect_detection(pattern, variations, ints, response, input_tag, response_tag, detected_departments)
                all_variations_correct = False

# End session logging with counters
with open(log_file_path, "a") as log_file:
    log_file.write(f"Session ended at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    log_file.write(f"Correct Variations: {correct_variations}\n")
    log_file.write(f"Incorrect Variations: {incorrect_variations}\n")
