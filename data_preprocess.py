import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pandas as pd

def preprocess_text(text):
    
    # Convert text to lowercase
    text = text.lower()

    # Remove non-alphabetic and non-space characters, preserving question marks for detection
    text = re.sub(r'[^a-z\s?]', '', text)
 
    # Tokenize text
    tokens = word_tokenize(text)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]

    # Lemmatization (reducing to root form)
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]

    # Detect question pattern
    #is_question = '?' in text or any(token in ['what', 'who', 'where', 'when', 'why', 'how'] for token in tokens)

    # Join tokens to recreate the sentence
    cleaned_text = ' '.join(lemmatized_tokens)

    return cleaned_text     #, is_question

def apply_preprocessing(csv_file):
    # Applying preprocessing to each sentence in the DataFrame
    data_frame = pd.read_csv(csv_file)
    results = data_frame['sentence'].apply(preprocess_text)

    # Extracting cleaned text and is_question flag into separate columns
    
    data_dict = {}
    data_dict['cleaned_text'] = results.apply(lambda x: x[0]) # result used as an arguemnt in lambda via apply
    data_dict['is_question'] = results.apply(lambda x: x[1])

    return pd.DataFrame(data=data_dict)