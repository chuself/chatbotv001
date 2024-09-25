import json
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
import pickle

# Load the config file for settings
with open('config.json', 'r') as config_file:
    config = json.load(config_file)

# Load intents from the file specified in the config
with open(config['intents_file'], 'r') as file:
    intents = json.load(file)

# Initialize lists for patterns and tags
patterns = []
tags = []

# Loop through each intent and extract patterns and tags, including key phrases from config
for intent in intents['intents']:
    for pattern in intent['patterns']:
        patterns.append(pattern)
        tags.append(intent['tag'])

# Include key phrases from config into patterns and tags
for department, data in config['key_phrases'].items():
    for sub_department, phrases in data['sub_departments'].items():
        for phrase in phrases:
            patterns.append(phrase)
            tags.append(sub_department)

# Print the extracted patterns and tags for debugging
print(f"Extracted {len(patterns)} patterns and {len(tags)} tags, including key phrases.")

# Initialize the tokenizer with settings from config
tokenizer = Tokenizer(num_words=config.get('num_words', 2000), oov_token=config.get('oov_token', "<OOV>"))
tokenizer.fit_on_texts(patterns)

# Save the tokenizer to a file
with open(config['tokenizer_file'], 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

print(f"Tokenizer has been created and saved as '{config['tokenizer_file']}'.")

# Initialize the label encoder and fit it to the tags
lbl_encoder = LabelEncoder()
lbl_encoder.fit(tags)

# Save the label encoder to a file
with open(config['label_encoder_file'], 'wb') as enc_file:
    pickle.dump(lbl_encoder, enc_file, protocol=pickle.HIGHEST_PROTOCOL)

print(f"Label encoder has been created and saved as '{config['label_encoder_file']}'.")
