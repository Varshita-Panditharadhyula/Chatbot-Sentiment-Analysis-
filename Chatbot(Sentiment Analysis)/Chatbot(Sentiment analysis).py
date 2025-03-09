# 1. Importing Libraries
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import pandas as pd

# Download required NLTK data files
nltk.download('stopwords')
nltk.download('punkt')

# 2. Data Preprocessing
# Larger example data
data = pd.DataFrame({
    'text': [
        'I love this product!',
        'This is the worst experience I have ever had.',
        'It was okay, nothing special.',
        'Absolutely fantastic!',
        'Horrible customer service!',
        'I am extremely happy with the service.',
        'The product is terrible and broke immediately.',
        'This is just average, not bad but not great.',
        'Amazing quality and fast delivery!',
        'Awful experience, would not recommend.',
        'Good value for the price.',
        'Terrible customer support, very disappointing.',
        'Superb! Will buy again.',
        'Meh, it’s fine but not what I expected.',
        'Loved it! Totally worth it.',
        'This is the worst thing I’ve ever purchased.',
        'The support team was helpful and quick to respond.',
        'It’s okay, nothing special but it works.'
    ],
    'sentiment': [
        'positive', 'negative', 'neutral', 'positive', 'negative',
        'positive', 'negative', 'neutral', 'positive', 'negative',
        'positive', 'negative', 'positive', 'neutral', 'positive',
        'negative', 'positive', 'neutral'
    ]
})

# Initialize stopwords
stop_words = set(stopwords.words('english'))

# Preprocessing function
def preprocess_text(text):
    try:
        words = word_tokenize(text.lower())  # Tokenize and convert to lowercase
        words = [word for word in words if word.isalnum() and word not in stop_words]  # Remove stopwords and non-alphanumeric
        return ' '.join(words)
    except Exception as e:
        print(f"Error during preprocessing: {e}")
        return ""

# Clean the data
data['cleaned_text'] = data['text'].apply(preprocess_text)

# Vectorization
try:
    vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))  # Allow bigrams for better context
    X = vectorizer.fit_transform(data['cleaned_text'])
    y = data['sentiment']
except ValueError as e:
    print(f"Vectorization error: {e}")

# Train-test split (stratified to balance classes)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Training the Model
try:
    model = MultinomialNB()
    model.fit(X_train, y_train)
except Exception as e:
    print(f"Training error: {e}")

# Save the model and vectorizer
try:
    joblib.dump(model, 'sentiment_model.pkl')
    joblib.dump(vectorizer, 'vectorizer.pkl')
except Exception as e:
    print(f"Error saving model: {e}")

# Evaluate the model
try:
    y_pred = model.predict(X_test)
    print("\nModel Evaluation:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
except Exception as e:
    print(f"Evaluation error: {e}")

# Load the model and vectorizer (optional, just to simulate production)
try:
    model = joblib.load('sentiment_model.pkl')
    vectorizer = joblib.load('vectorizer.pkl')
except Exception as e:
    print(f"Error loading model: {e}")

# 4. Interactive Chatbot
def chatbot():
    print("🤖 Chatbot is ready! Type 'quit' to exit.\n")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'quit':
            print("👋 Goodbye!")
            break
        
        # Preprocess input
        cleaned_input = preprocess_text(user_input)
        if not cleaned_input:
            print("Bot: Sorry, I didn't catch that. Can you try again? 🤔\n")
            continue
        
        # Vectorize input and predict sentiment
        try:
            vectorized_input = vectorizer.transform([cleaned_input])
            sentiment = model.predict(vectorized_input)[0]
        except Exception as e:
            print(f"Prediction error: {e}")
            continue
        
        # Generate response based on sentiment
        if sentiment == 'positive':
            response = "I'm glad to hear that! 😊"
        elif sentiment == 'negative':
            response = "I'm sorry to hear that. 😔"
        elif sentiment == 'neutral':
            response = "I see. 🤔"
        else:
            response = "Hmm, not sure how to respond to that. 🤔"
        
        print(f"Bot: {response}\n")

# Start chatbot
chatbot()