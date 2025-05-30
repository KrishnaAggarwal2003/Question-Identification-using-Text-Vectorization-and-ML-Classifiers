# Question-Identification-using-Text-Vectorization-and-ML-Classifiers
This project focuses on identifying whether a given sentence is a question using various text vectorization techniques and classical machine learning classifiers. It preprocesses text data, transforms it using methods like TF-IDF, word embeddings, LDA, and LSA, and evaluates the performance of ML classifiers such as Logistic Regression, Naive Bayes, Random Forest, and SVM. The results help determine the most effective combination for question identification in text.

## Project Structure

```
.
├── main.ipynb                # Main notebook to run experiments and visualize results
├── vectorization_models.py   # Implements different text vectorization methods
├── classification_models.py  # Contains classification and ROC plotting logic
├── data_preprocess.py        # Text cleaning and preprocessing functions
├── Project_data.csv          # Text dataset
```

## Data Preprocessing

- **Lowercasing:** Converts all text to lowercase.
- **Cleaning:** Removes non-alphabetic characters (except question marks).
- **Tokenization:** Splits sentences into words.
- **Stopword Removal:** Removes common English stopwords.
- **Lemmatization:** Reduces words to their root form.

All preprocessing is handled by `data_preprocess.py`.

## Text Vectorization Models

Implemented in `vectorization_models.py`:

1. **TF-IDF Text Vectorization:**  
   Converts text to TF-IDF features.
2. **One-word Embedding:**  
   Uses Keras Embedding and GlobalAveragePooling1D to generate dense representations.
3. **LDA Text Classification:**  
   Applies Latent Dirichlet Allocation for topic-based features.
4. **LSA Text Vectorization:**  
   Uses Truncated SVD (LSA) on TF-IDF features for dimensionality reduction.


## Classification Models

Implemented in `classification_models.py` and evaluated using ROC curves:

- Logistic Regression
- Bernoulli Naive Bayes
- Random Forest
- Support Vector Machine (SVM)

## How to Run

1. **Install dependencies:**
   ```
   pip install pandas scikit-learn nltk tensorflow matplotlib
   ```
2. **Download NLTK resources:**
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   nltk.download('wordnet')
   ```
3. **Run the notebook:**  
   Open `main.ipynb` in VS Code or Jupyter and execute all cells.

## Results   

For each vectorization method, the following classifiers were evaluated: Logistic Regression, Bernoulli Naive Bayes, Random Forest, and SVM. The ROC curves for each combination are plotted for visual comparison.

### Performance Plots
- **TFI-DF Text Vectorization**

![image](https://github.com/user-attachments/assets/3f852205-2f5b-4ab6-bbb9-ca4d600b1a24)

- **One-word Embedding**

![image](https://github.com/user-attachments/assets/267ba4b4-823b-419b-9fa2-efb0947fe230)

- **LDA Text Classification**

![image](https://github.com/user-attachments/assets/cf18cca2-0ded-4399-8c69-a4245a028a3c)

- **LSA Text Vectorization**

![image](https://github.com/user-attachments/assets/eaedb6b1-d5e6-4058-9374-996a4bc2632e)


## Conclusion

- The project demonstrates the effectiveness of combining text vectorization with classical classifiers for question identification.
- The best performance was achieved using **[One-word embedding + Random Forest]** with the ROC curve area of 86%.
- **Random Forest** consistently outperformed other classifiers, achieving the highest ROC curve values across all vectorization models. This demonstrates its effectiveness for question identification in text using the tested approaches.









