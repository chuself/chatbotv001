# # import os #  Provides a way to interact with the operating system, such as changing directories.
# # print(os.getcwd())#Prints the current working directory to verify where you are.
# # os.chdir(r'd:\Programing\Projects\ChatBotV1')  # Using raw string for Windows file path
# # # Changes the current working directory to the specified path where your project files are located.

# # import random #Allows you to perform operations related to randomness, like shuffling lists.
# # import json #Provides methods to work with JSON data, such as loading and parsing JSON files.
# # import pickle # Used for serializing and deserializing Python objects, like saving and loading processed data. Save and load files
# # import numpy as np #Provides support for large arrays and matrices, and mathematical functions to operate on these arrays.
# # import datetime


# # # import sys
# # # sys.stdout = open('training_log.txt', 'w')
# # # model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=10, verbose=0)
# # # sys.stdout.close()

# # import nltk #The Natural Language Toolkit, used for natural language processing tasks.
# # nltk.download('punkt') #Downloads the tokenizer models needed for tokenizing text.
# # nltk.download('wordnet') #Downloads the WordNet lexical database used for lemmatization.
# # nltk.download('stopwords')
# # from nltk.stem import WordNetLemmatizer #Imports the WordNetLemmatizer, which reduces words to their base or root form.

# # from tensorflow.keras.models import Sequential #Imports the Sequential model from TensorFlow’s Keras API, which is a linear stack of layers.
# # from tensorflow.keras.layers import Dense, Activation, Dropout# Imports Dense and Dropout layers, which are types of layers used in neural networks.
# # from tensorflow.keras.optimizers import SGD #Imports the Stochastic Gradient Descent optimizer, used for training the neural network.

# # from tensorflow.keras.regularizers import l2
# # from tensorflow.keras.layers import BatchNormalization

# # #Setup and Data Preparation
# # lemmatizer = WordNetLemmatizer() #Creates an instance of the WordNetLemmatizer, which will be used to reduce words to their base form.

# # # Load and Process Intents
# # with open('intents.json') as file:
# #     intents = json.load(file)#Reads and parses the intents.json file to extract the chatbot’s intent data.

# # words = []#words: List to hold all unique words from the training data.
# # classes = []#classes: List to hold all unique intent tags.
# # documents = []#documents: List to hold pairs of tokenized words and their corresponding intent tag.
# # ignore_letters = ['?', ',', '.', '!']#ignore_letters: List of punctuation marks to be ignored during tokenization.

# # # Preprocessing the input data
# # for intent in intents['intents']:#
# #     for pattern in intent['patterns']:#
# #         word_list = nltk.word_tokenize(pattern)#Tokenizes each pattern into individual words
# #         words.extend(word_list)# Adds the tokenized words to the words list.
# #         documents.append((word_list, intent['tag']))# Appends a tuple of tokenized words and their intent tag to documents.
# #         if intent['tag'] not in classes:#Checks if the tag is not already in the classes list and adds it if it's not.
# #             classes.append(intent['tag'])#add the tag if not in class list

# # # Lemmatizing the words and filtering
# # words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in ignore_letters]#Lemmatizes each word and converts it to lowercase.
# # words = sorted(set(words))#Removes duplicates and sorts the words alphabetically.

# # # Sorting the classes
# # classes = sorted(set(classes))#

# # # Saving processed words and classes to pickle files
# # pickle.dump(words, open('words.pkl', 'wb'))# Saves the processed words list to a file.
# # pickle.dump(classes, open('classes.pkl', 'wb'))#Saves the classes list to a file.

# # # Training data preparation
# # training = []# creates A variable and assins it an empty list
# # output_empty = [0] * len(classes)#Creates a zeroed vector for the output classes.

# # for document in documents:#
# #     bag = []#
# #     word_patterns = document[0]  # The list of tokenized words in the pattern
# #     word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]#
    
# #     # Create the bag of words
# #     for word in words:#
# #         bag.append(1) if word in word_patterns else bag.append(0)#Creates a bag-of-words vector where 1 indicates the presence of a word and 0 indicates absence.

# #     # Create the output row (one-hot encoded for the corresponding tag)
# #     output_row = list(output_empty)#
# #     output_row[classes.index(document[1])] = 1#Sets the appropriate index in the output vector to 1, indicating the correct intent.
    
# #     # Append the training data
# #     training.append([bag, output_row])#

# # # Shuffle the training data
# # random.shuffle(training)# Shuffles the training data to ensure the model doesn't learn any unintended order.

# # # Convert training data into NumPy arrays
# # training = np.array(training, dtype=object)#Converts the training data into a NumPy array.

# # # Separate the training data into features and labels
# # train_x = np.array([element[0] for element in training])# Separate features and labels for training
# # train_y = np.array([element[1] for element in training])# Separate features and labels for training

# # model = Sequential()
# # model.add(Dense(256, input_shape=(len(train_x[0]),), activation='relu', kernel_regularizer=l2(0.001)))
# # model.add(BatchNormalization())
# # model.add(Dropout(0.4))

# # model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.001)))
# # model.add(BatchNormalization())
# # model.add(Dropout(0.4))

# # model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.001)))
# # model.add(BatchNormalization())
# # model.add(Dropout(0.4))

# # model.add(Dense(len(train_y[0]), activation='softmax'))



# # # model = Sequential()#
# # # model.add(Dense(128, input_shape=(len(train_x[0]), ), activation = 'relu'))#
# # # model.add(Dropout(0.5))#
# # # model.add(Dense(64, activation='relu'))#
# # # model.add(Dropout(0.5))#
# # # model.add(Dense(len(train_y[0]), activation='softmax'))#


# # # sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)#
# # # model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])#

# # from nltk.corpus import stopwords
# # stop_words = set(stopwords.words('english'))

# # def remove_stopwords(sentence):
# #     words = nltk.word_tokenize(sentence)
# #     filtered_sentence = [word for word in words if word not in stop_words]
# #     return ' '.join(filtered_sentence)

# # augmented_patterns = []

# # for intent in intents['intents']:
# #     for pattern in intent['patterns']:
# #         # Original pattern
# #         augmented_patterns.append(pattern)
        
# #         # Apply augmentation techniques
# #         augmented_patterns.append(synonym_replacement(pattern))
# #         augmented_patterns.append(random_deletion(pattern))
# #         augmented_patterns.append(random_swap(pattern))
# #         augmented_patterns.append(remove_stopwords(pattern))

# #     # Add the new patterns to the intents
# #     intent['patterns'] = augmented_patterns



# # from tensorflow.keras.optimizers import Adam
# # model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])


# # from tensorflow.keras.callbacks import EarlyStopping# Add EarlyStopping Example:
# # early_stopping = EarlyStopping(monitor='loss', patience=10, verbose=1)

# # from tensorflow.keras.callbacks import ReduceLROnPlateau
# # lr_scheduler = ReduceLROnPlateau(monitor='loss', factor=0.2, patience=3, verbose=1)

# # from tensorflow.keras.callbacks import TensorBoard
# # log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# # tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=1)

# # hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=10, verbose=1,validation_split=0.2, callbacks=[early_stopping, lr_scheduler, tensorboard])#
# # model.save('chatbotmodel.h5')#

# # print("Training data successfully processed.")#

# import os  # Provides a way to interact with the operating system, such as changing directories.
# import random  # Allows you to perform operations related to randomness, like shuffling lists.
# import json  # Provides methods to work with JSON data, such as loading and parsing JSON files.
# import pickle  # Used for serializing and deserializing Python objects, like saving and loading processed data.
# import numpy as np  # Provides support for large arrays and matrices, and mathematical functions to operate on these arrays.
# import datetime
# import nltk  # The Natural Language Toolkit, used for natural language processing tasks.

# from nltk.stem import WordNetLemmatizer  # Reduces words to their base or root form.
# from nltk.corpus import stopwords
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Activation, Dropout, BatchNormalization
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard
# from tensorflow.keras.regularizers import l2

# # Setting up directory and verifying path
# try:
#     os.chdir(r'd:\Programing\Projects\ChatBotV1')
#     print(f"Changed directory to {os.getcwd()}")
# except FileNotFoundError:
#     print("Directory not found. Please check the file path.")
#     raise

# # Download NLTK data (tokenizers and stopwords)
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('stopwords')

# lemmatizer = WordNetLemmatizer()  # Instance of WordNetLemmatizer

# # Load and process intents
# with open('intents.json') as file:
#     intents = json.load(file)

# words = []
# classes = []
# documents = []
# ignore_letters = ['?', ',', '.', '!']

# # Preprocessing the input data
# for intent in intents['intents']:
#     for pattern in intent['patterns']:
#         word_list = nltk.word_tokenize(pattern)
#         words.extend(word_list)
#         documents.append((word_list, intent['tag']))
#         if intent['tag'] not in classes:
#             classes.append(intent['tag'])

# # Lemmatizing the words and filtering
# words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in ignore_letters]
# words = sorted(set(words))

# # Sorting the classes
# classes = sorted(set(classes))

# # Saving processed words and classes to pickle files
# pickle.dump(words, open('words.pkl', 'wb'))
# pickle.dump(classes, open('classes.pkl', 'wb'))

# # Training data preparation
# training = []
# output_empty = [0] * len(classes)

# for document in documents:
#     bag = []
#     word_patterns = document[0]
#     word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    
#     for word in words:
#         bag.append(1) if word in word_patterns else bag.append(0)
    
#     output_row = list(output_empty)
#     output_row[classes.index(document[1])] = 1
#     training.append([bag, output_row])

# random.shuffle(training)
# training = np.array(training, dtype=object)

# # Separate the training data into features and labels
# train_x = np.array([element[0] for element in training])
# train_y = np.array([element[1] for element in training])

# # Defining the Neural Network model with reduced dropout rate
# model = Sequential()
# model.add(Dense(256, input_shape=(len(train_x[0]),), activation='relu', kernel_regularizer=l2(0.001)))
# model.add(BatchNormalization())
# model.add(Dropout(0.4))

# model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.001)))
# model.add(BatchNormalization())
# model.add(Dropout(0.4))

# model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.001)))
# model.add(BatchNormalization())
# model.add(Dropout(0.4))

# model.add(Dense(len(train_y[0]), activation='softmax'))

# # Compile the model
# model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

# # Define stopwords removal function
# stop_words = set(stopwords.words('english'))

# def remove_stopwords(sentence):
#     words = nltk.word_tokenize(sentence)
#     filtered_sentence = [word for word in words if word not in stop_words]
#     return ' '.join(filtered_sentence)

# # Data augmentation functions
# def synonym_replacement(sentence):
#     # Replace words with their synonyms (this is a placeholder, replace with your logic)
#     words = nltk.word_tokenize(sentence)
#     for i, word in enumerate(words):
#         if random.random() < 0.3:  # Replace 30% of the words
#             words[i] = lemmatizer.lemmatize(word)  # Simplified: Use lemmatized form as "synonym"
#     return ' '.join(words)

# def random_deletion(sentence, p=0.2):
#     # Randomly delete words from the sentence with probability p
#     words = nltk.word_tokenize(sentence)
#     if len(words) == 1:  # Don't delete if it's the only word
#         return sentence
#     return ' '.join([word for word in words if random.random() > p])

# def random_swap(sentence, n=1):
#     # Tokenize the sentence into words
#     words = nltk.word_tokenize(sentence)
    
#     # If there are fewer than 2 words, return the sentence as is (cannot swap)
#     if len(words) < 2:
#         return sentence
    
#     # Perform n random swaps
#     for _ in range(n):
#         idx1, idx2 = random.sample(range(len(words)), 2)  # Randomly pick two indices
#         words[idx1], words[idx2] = words[idx2], words[idx1]  # Swap the two words
    
#     return ' '.join(words)


# # Augment patterns with the new data
# augmented_patterns = []
# for intent in intents['intents']:
#     temp_patterns = []
#     for pattern in intent['patterns']:
#         temp_patterns.append(pattern)
#         temp_patterns.append(synonym_replacement(pattern))
#         temp_patterns.append(random_deletion(pattern))
#         temp_patterns.append(random_swap(pattern))
#         temp_patterns.append(remove_stopwords(pattern))
#     intent['patterns'] = temp_patterns
#     augmented_patterns.extend(temp_patterns)

# # Callbacks for training
# early_stopping = EarlyStopping(monitor='loss', patience=10, verbose=1)
# lr_scheduler = ReduceLROnPlateau(monitor='loss', factor=0.2, patience=3, verbose=1)
# log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=1)

# # Train the model
# hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=10, verbose=1, 
#                  validation_split=0.2, callbacks=[early_stopping, lr_scheduler, tensorboard])

# # Save the trained model
# model.save('chatbotmodel.h5')

# print("Training data successfully processed.")





















# import os  # Provides a way to interact with the operating system, such as changing directories.
# import random  # Allows you to perform operations related to randomness, like shuffling lists.
# import json  # Provides methods to work with JSON data, such as loading and parsing JSON files.
# import pickle  # Used for serializing and deserializing Python objects, like saving and loading processed data.
# import numpy as np  # Provides support for large arrays and matrices, and mathematical functions to operate on these arrays.
# import datetime
# import nltk  # The Natural Language Toolkit, used for natural language processing tasks.
# import matplotlib
# matplotlib.use('Agg')  # Use a non-GUI backend
# import matplotlib.pyplot as plt

# from nltk.stem import WordNetLemmatizer  # Reduces words to their base or root form.
# from nltk.corpus import stopwords
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard
# from tensorflow.keras.regularizers import l2
# from tensorflow.keras.optimizers import SGD #Imports the Stochastic Gradient Descent optimizer, used for training the neural network
# # Setting up directory and verifying path
# from tensorflow.keras.layers import LeakyReLU

# try:
#     os.chdir(r'd:\Programing\Projects\ChatBotV1')  # Ensure this path is correct for your environment
#     print(f"Changed directory to {os.getcwd()}")
# except FileNotFoundError:
#     print("Directory not found. Please check the file path.")
#     raise

# #Download NLTK data (tokenizers and stopwords)
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('stopwords')

# lemmatizer = WordNetLemmatizer()  # Instance of WordNetLemmatizer

# # Load and process intents
# with open('dep_intents.json') as file:
#     intents = json.load(file)

# words = []
# classes = []
# documents = []
# ignore_letters = ['?', ',', '.', '!']

# # Preprocessing the input data
# for intent in intents['intents']:
#     for pattern in intent['patterns']:
#         word_list = nltk.word_tokenize(pattern)
#         words.extend(word_list)
#         documents.append((word_list, intent['tag']))
#         if intent['tag'] not in classes:
#             classes.append(intent['tag'])

# # Lemmatizing the words and filtering
# words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in ignore_letters]
# words = sorted(set(words))

# # Sorting the classes
# classes = sorted(set(classes))

# # Saving processed words and classes to pickle files
# pickle.dump(words, open('words.pkl', 'wb'))
# pickle.dump(classes, open('classes.pkl', 'wb'))

# # Training data preparation
# training = []
# output_empty = [0] * len(classes)

# for document in documents:
#     bag = []
#     word_patterns = document[0]
#     word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    
#     for word in words:
#         bag.append(1) if word in word_patterns else bag.append(0)
    
#     output_row = list(output_empty)
#     output_row[classes.index(document[1])] = 1
#     training.append([bag, output_row])

# random.shuffle(training)
# training = np.array(training, dtype=object)

# # Separate the training data into features and labels
# # train_x = np.array([element[0] for element in training])
# # train_y = np.array([element[1] for element in training])
# train_x = list(training[:, 0])
# train_y = list(training[:, 1])


# # Defining the Neural Network model
# model = Sequential()

# # First layer with input shape
# model.add(Dense(256, input_shape=(len(train_x[0]),), activation='relu', kernel_regularizer=l2(0.001)))
# model.add(BatchNormalization())
# model.add(Dropout(0.4))

# # Second layer
# model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.001)))
# model.add(BatchNormalization())
# model.add(Dropout(0.4))

# # Third layer
# model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.001)))
# model.add(BatchNormalization())
# model.add(Dropout(0.4))

# # Output layer
# model.add(Dense(len(train_y[0]), activation='softmax'))


# # Compile the model
# # model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
# sgd = SGD (learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
# model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])


# # # Define stopwords removal function
# # stop_words = set(stopwords.words('english'))

# # def remove_stopwords(sentence):
# #     words = nltk.word_tokenize(sentence)
# #     filtered_sentence = [word for word in words if word not in stop_words]
# #     return ' '.join(filtered_sentence)

# # # Data augmentation functions
# # def synonym_replacement(sentence):
# #     words = nltk.word_tokenize(sentence)
# #     for i, word in enumerate(words):
# #         if random.random() < 0.3:  # Replace 30% of the words
# #             words[i] = lemmatizer.lemmatize(word)  # Simplified: Use lemmatized form as "synonym"
# #     return ' '.join(words)

# # def random_deletion(sentence, p=0.2):
# #     words = nltk.word_tokenize(sentence)
# #     if len(words) == 1:  # Don't delete if it's the only word
# #         return sentence
# #     return ' '.join([word for word in words if random.random() > p])

# # def random_swap(sentence, n=1):
# #     words = nltk.word_tokenize(sentence)
# #     if len(words) < 2:
# #         return sentence
# #     for _ in range(n):
# #         idx1, idx2 = random.sample(range(len(words)), 2)
# #         words[idx1], words[idx2] = words[idx2], words[idx1]
# #     return ' '.join(words)

# # def controlled_augmentation(sentence, p_deletion=0.2, n_swap=1):
# #     # Apply random deletion
# #     sentence = random_deletion(sentence, p_deletion)
    
# #     # Apply random swap
# #     sentence = random_swap(sentence, n_swap)
    
# #     return sentence



# # # Augment patterns with the new data
# # augmented_patterns = []
# # for intent in intents['intents']:
# #     temp_patterns = []
# #     for pattern in intent['patterns']:
# #         temp_patterns.append(pattern)
# #         temp_patterns.append(synonym_replacement(pattern))
# #         temp_patterns.append(random_deletion(pattern))
# #         temp_patterns.append(random_swap(pattern))
# #         temp_patterns.append(remove_stopwords(pattern))
# #     intent['patterns'] = temp_patterns
# #     augmented_patterns.extend(temp_patterns)

# # # Callbacks for training
# # early_stopping = EarlyStopping(monitor='loss', patience=10, verbose=1)
# # lr_scheduler = ReduceLROnPlateau(monitor='loss', factor=0.2, patience=3, verbose=1)
# # log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# # tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=1)

# # Train the model
# hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)# validation_split=0.2, callbacks=[early_stopping, lr_scheduler, tensorboard])

# ## Plot training and validation loss


# # plt.plot(hist.history['loss'], label='loss')
# # plt.plot(hist.history['val_loss'], label = 'val_loss')
# # plt.xlabel('Epoch')
# # plt.ylabel('Loss')
# # plt.legend()
# # plt.show()
# # plt.savefig('plotfileimage.png')

# # Save the trained model
# model.save('chatbotmodel.keras', hist)

# print("Training data successfully processed.")











import os
import random
import json
import pickle
import numpy as np
import datetime
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import SGD

# Setting up directory and verifying path
try:
    os.chdir(r'd:\Programing\Projects\ChatBotV1')
    print(f"Changed directory to {os.getcwd()}")
except FileNotFoundError:
    print("Directory not found. Please check the file path.")
    raise

# Download NLTK data
nltk.download('punkt')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

# Load key phrases from config.json
with open('config.json') as config_file:
    config = json.load(config_file)

# Load and process intents
with open('dep_intents.json') as file:
    intents = json.load(file)

words = []
classes = []
documents = []
ignore_letters = ['?', ',', '.', '!']

# Preprocessing the input data
for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent['tag']))

# Add key phrases to the words list
for category in config['key_phrases']:
    for sub_dept in config['key_phrases'][category]['sub_departments']:
        words.extend(config['key_phrases'][category]['sub_departments'][sub_dept])

# Filter and lemmatize words
words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in ignore_letters]
words = sorted(set(words))

# Sorting the classes
classes = sorted(set(intent['tag'] for intent in intents['intents']))

# Saving processed words and classes to pickle files
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# Training data preparation
training = []
output_empty = [0] * len(classes)

for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]

    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append([bag, output_row])

random.shuffle(training)
training = np.array(training, dtype=object)

# Separate the training data into features and labels
train_x = list(training[:, 0])
train_y = list(training[:, 1])

# Defining the Neural Network model
model = Sequential()
model.add(Dense(256, input_shape=(len(train_x[0]),), activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))
model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))
model.add(Dense(64, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Compile the model
sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Train the model
hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)

# Save the trained model
model.save('chatbotmodel.keras', hist)

print("Training data successfully processed.")
