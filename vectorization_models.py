from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from data_preprocess import preprocess_text
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, Embedding, GlobalAveragePooling1D
from tensorflow.keras.models import Model
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD


def text_models(x, y, model_name, test_size=0.3):
    print(f'Using the model: {model_name}')
    
    # TF-IDF Text Vectorization 
    if model_name == 'TF-IDF Text Vectorization':
        tfidf_vectorizer = TfidfVectorizer(preprocessor=preprocess_text)
        tfidf_matrix = tfidf_vectorizer.fit_transform(x)
        x_train, x_test, y_train, y_test = train_test_split(tfidf_matrix, y, test_size=test_size, random_state=42)
        return x_train, x_test, y_train, y_test

    # One-word Embedding
    elif model_name == 'One-word Embedding':
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(x)
        sequences = tokenizer.texts_to_sequences(x)
        max_length = max(len(seq) for seq in sequences)
        X_padded = pad_sequences(sequences, maxlen=max_length, padding='post')
        
        input_layer = Input(shape=(max_length,))
        embedding_layer = Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=50, input_length=max_length)(input_layer)
        pooling_layer = GlobalAveragePooling1D()(embedding_layer)
        embedding_model = Model(inputs=input_layer, outputs=pooling_layer)
        
        # Extract embeddings
        X_embeddings = embedding_model.predict(X_padded)

    # Split the data into training and testing sets
        x_train, x_test, y_train, y_test = train_test_split(X_embeddings, y, test_size=test_size, random_state=42)
        return x_train, x_test, y_train, y_test      

    # LDA Text Classification
    elif model_name == 'LDA Text Classification':
        vectorizer = CountVectorizer(preprocessor=preprocess_text,analyzer='word', min_df=5, stop_words='english', lowercase=True, token_pattern='[a-zA-Z0-9]{3,}')
        data_vectorized = vectorizer.fit_transform(x)
        lda = LatentDirichletAllocation(n_components=50, random_state=100)
        X_data_at = lda.fit_transform(data_vectorized)
        x_train, x_test, y_train, y_test = train_test_split(X_data_at, y, test_size=test_size, random_state=42)
        return x_train, x_test, y_train, y_test

    # LSA Text Vectorization
    elif model_name == 'LSA Text Vectorization':
        tfidf_vectorizer = TfidfVectorizer(preprocessor=preprocess_text)
        tfidf_embeddings = tfidf_vectorizer.fit_transform(x)
        svd = TruncatedSVD(n_components=10, random_state=42)
        lsa_embeddings = svd.fit_transform(tfidf_embeddings)
        x_train, x_test, y_train, y_test = train_test_split(lsa_embeddings, y, test_size=test_size, random_state=42)
        return x_train, x_test, y_train, y_test
    
    else:
        raise ValueError(f"Model '{model_name}' not recognized!")
        