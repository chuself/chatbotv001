import nltk

# Download the 'punkt' tokenizer
nltk.download('punkt_tab', download_dir='/opt/render/nltk_data')



# Specify the directory where you want to download NLTK data
nltk_data_dir = 'D:/Programing/Projects/ChatBotV1/nltk_data'

# Add the path to the NLTK data search path
nltk.data.path.append(nltk_data_dir)

# Download the 'punkt' package to the specified directory
nltk.download('punkt_tab', download_dir=nltk_data_dir)
