import json
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
import pickle

# Load intents from the JSON file
with open('intents.json') as file:
    intents = json.load(file)

# Initialize lists for patterns and tags
patterns = []
tags = []

# Loop through each intent and extract patterns and tags
for intent in intents['intents']:
    for pattern in intent['patterns']:
        patterns.append(pattern)
        tags.append(intent['tag'])

# Print the extracted patterns and tags for debugging
print(f"Extracted {len(patterns)} patterns and {len(tags)} tags.")

# Initialize the tokenizer
tokenizer = Tokenizer(num_words=2000, oov_token="<OOV>")
tokenizer.fit_on_texts(patterns)

# Save the tokenizer to a file
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("Tokenizer has been created and saved as 'tokenizer.pickle'.")

# Initialize the label encoder and fit it to the tags
lbl_encoder = LabelEncoder()
lbl_encoder.fit(tags)

# Save the label encoder to a file
with open('label_encoder.pickle', 'wb') as enc_file:
    pickle.dump(lbl_encoder, enc_file, protocol=pickle.HIGHEST_PROTOCOL)

print("Label encoder has been created and saved as 'label_encoder.pickle'.")
