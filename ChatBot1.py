import os
print(os.getcwd())
os.chdir('d:\Programing\Projects\ChatBotV1')

import os
import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
import difflib  # For fuzzy matching

lemmatizer = WordNetLemmatizer()

try:
    intents = json.loads(open('intents.json').read())
    words = pickle.load(open('words.pkl', 'rb'))
    classes = pickle.load(open('classes.pkl', 'rb'))
    model = load_model('chatbotmodel.h5')
except Exception as e:
    print(f"Error loading files or model: {str(e)}")
    exit()

# Initialize context management
context = {}

# Fuzzy matching function
def fuzzy_match(user_input, known_words):
    patterns = []
    for intent in intents['intents']:
        patterns.extend(intent['patterns'])
    return difflib.get_close_matches(user_input, known_words, n=1, cutoff=0.7)

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
    ERROR_THRESHOLD = 0.75
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    if not results:
        return [{'intent': 'fallback', 'probability': '1.0'}]  # Return fallback if no matches

    results.sort(key=lambda x: x[1], reverse=True)
    return [{'intent': classes[r[0]], 'probability': str(r[1])} for r in results]

def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']

    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    
    # Update context if needed
    if tag == 'joke':
        context['context'] = 'joke_context'
    else:
        context['context'] = None

    # Add response variation here (change based on context, time, etc.)
    if context['context'] == 'joke_context':
        result = "Another joke? Here's a good one: " + result

    return result

# Validation of input
def is_input_valid(sentence_words):
    return any(word in words for word in sentence_words)

print("Nexus is Online")

while True:
    message = input("")

    if not message.strip():  # Skip if message is empty or just spaces
        print("Please enter something!")
        continue

    # First, check for fuzzy matches with known words
    fuzzy_matched_word = fuzzy_match(message, words)
    if fuzzy_matched_word:
        message = fuzzy_matched_word[0]  # Update the message with the closest match
    else:
        # If no fuzzy match is found, notify the user or handle it
        print("Sorry, I couldn't understand that.")
        continue
    # sentence_words = clean_up_sentence(message)

    # # Validate the input after fuzzy matching
    # if not is_input_valid(sentence_words):
    #     print("Sorry, I couldn't understand that.")
    #     continue

    ints = predict_class(message)

    # If the top intent is fallback, provide a default response
    if ints[0]['intent'] == 'fallback':
        print("Sorry, I don't understand. Could you rephrase that?")
        continue

    res = get_response(ints, intents)

    print(res)
    
    # Reduced debugging info (optional: you can uncomment these for full debugging)
    print(f"Predicted intents: {ints}")
    print(f"Current context: {context}")
    print(f"Model prediction: {model.predict(np.array([bag_of_words(message)]))}")

    

 