import os #  Provides a way to interact with the operating system, such as changing directories.
print(os.getcwd())#Prints the current working directory to verify where you are.
os.chdir(r'd:\Programing\Projects\ChatBotV1')  # Using raw string for Windows file path
# Changes the current working directory to the specified path where your project files are located.

import random #Allows you to perform operations related to randomness, like shuffling lists.
import json #Provides methods to work with JSON data, such as loading and parsing JSON files.
import pickle # Used for serializing and deserializing Python objects, like saving and loading processed data. Save and load files
import numpy as np #Provides support for large arrays and matrices, and mathematical functions to operate on these arrays.
import datetime


# import sys
# sys.stdout = open('training_log.txt', 'w')
# model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=10, verbose=0)
# sys.stdout.close()

import nltk #The Natural Language Toolkit, used for natural language processing tasks.
nltk.download('punkt') #Downloads the tokenizer models needed for tokenizing text.
nltk.download('wordnet') #Downloads the WordNet lexical database used for lemmatization.
from nltk.stem import WordNetLemmatizer #Imports the WordNetLemmatizer, which reduces words to their base or root form.

from tensorflow.keras.models import Sequential #Imports the Sequential model from TensorFlow’s Keras API, which is a linear stack of layers.
from tensorflow.keras.layers import Dense, Activation, Dropout# Imports Dense and Dropout layers, which are types of layers used in neural networks.
from tensorflow.keras.optimizers import SGD #Imports the Stochastic Gradient Descent optimizer, used for training the neural network.

#Setup and Data Preparation
lemmatizer = WordNetLemmatizer() #Creates an instance of the WordNetLemmatizer, which will be used to reduce words to their base form.

# Load and Process Intents
with open('intents.json') as file:
    intents = json.load(file)#Reads and parses the intents.json file to extract the chatbot’s intent data.

words = []#words: List to hold all unique words from the training data.
classes = []#classes: List to hold all unique intent tags.
documents = []#documents: List to hold pairs of tokenized words and their corresponding intent tag.
ignore_letters = ['?', ',', '.', '!']#ignore_letters: List of punctuation marks to be ignored during tokenization.

# Preprocessing the input data
for intent in intents['intents']:#
    for pattern in intent['patterns']:#
        word_list = nltk.word_tokenize(pattern)#Tokenizes each pattern into individual words
        words.extend(word_list)# Adds the tokenized words to the words list.
        documents.append((word_list, intent['tag']))# Appends a tuple of tokenized words and their intent tag to documents.
        if intent['tag'] not in classes:#Checks if the tag is not already in the classes list and adds it if it's not.
            classes.append(intent['tag'])#add the tag if not in class list

# Lemmatizing the words and filtering
words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in ignore_letters]#Lemmatizes each word and converts it to lowercase.
words = sorted(set(words))#Removes duplicates and sorts the words alphabetically.

# Sorting the classes
classes = sorted(set(classes))#

# Saving processed words and classes to pickle files
pickle.dump(words, open('words.pkl', 'wb'))# Saves the processed words list to a file.
pickle.dump(classes, open('classes.pkl', 'wb'))#Saves the classes list to a file.

# Training data preparation
training = []# creates A variable and assins it an empty list
output_empty = [0] * len(classes)#Creates a zeroed vector for the output classes.

for document in documents:#
    bag = []#
    word_patterns = document[0]  # The list of tokenized words in the pattern
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]#
    
    # Create the bag of words
    for word in words:#
        bag.append(1) if word in word_patterns else bag.append(0)#Creates a bag-of-words vector where 1 indicates the presence of a word and 0 indicates absence.

    # Create the output row (one-hot encoded for the corresponding tag)
    output_row = list(output_empty)#
    output_row[classes.index(document[1])] = 1#Sets the appropriate index in the output vector to 1, indicating the correct intent.
    
    # Append the training data
    training.append([bag, output_row])#

# Shuffle the training data
random.shuffle(training)# Shuffles the training data to ensure the model doesn't learn any unintended order.

# Convert training data into NumPy arrays
training = np.array(training, dtype=object)#Converts the training data into a NumPy array.

# Separate the training data into features and labels
train_x = np.array([element[0] for element in training])# Separate features and labels for training
train_y = np.array([element[1] for element in training])# Separate features and labels for training

model = Sequential()#
model.add(Dense(128, input_shape=(len(train_x[0]), ), activation = 'relu'))#
model.add(Dropout(0.5))#
model.add(Dense(64, activation='relu'))#
model.add(Dropout(0.5))#
model.add(Dense(len(train_y[0]), activation='softmax'))#


# sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)#
# model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])#

from tensorflow.keras.optimizers import Adam
model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])


from tensorflow.keras.callbacks import EarlyStopping# Add EarlyStopping Example:
early_stopping = EarlyStopping(monitor='loss', patience=10, verbose=1)

from tensorflow.keras.callbacks import ReduceLROnPlateau
lr_scheduler = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, verbose=1)

from tensorflow.keras.callbacks import TensorBoard
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=1)

hist = model.fit(np.array(train_x), np.array(train_y), epochs=150, batch_size=10, verbose=1,validation_split=0.2, callbacks=[early_stopping, lr_scheduler, tensorboard])#
model.save('chatbotmodel.h5')#

print("Training data successfully processed.")#
