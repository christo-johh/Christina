import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

# Download necessary NLTK data (run once)
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
stop_words = set(stopwords.words('english'))

def load_data(ratings_file, movies_file):
    ratings_df = pd.read_csv(ratings_file)
    movies_df = pd.read_csv(movies_file)
    return ratings_df, movies_df

def preprocess_data(ratings_df, movies_df):
    # Merge dataframes
    df = pd.merge(ratings_df, movies_df, on='movie_id', how='inner')

    # Handle missing values (example: drop rows with missing genre)
    df.dropna(subset=['genres'], inplace=True)

    # Feature engineering (example: year of release)
    df['release_year'] = pd.to_datetime(df['release_date'], errors='coerce').dt.year
    df.drop('release_date', axis=1, inplace=True)

    # Split data
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    return train_df, test_df

def get_content_features(df):
    # Preprocess text data (plot summaries - assuming 'overview' column exists)
    if 'overview' in df.columns:
        df['overview'] = df['overview'].fillna('')
        df['overview'] = df['overview'].apply(lambda x: ' '.join([w.lower() for w in word_tokenize(x) if w.lower() not in stop_words and w.isalnum()]))
        tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf_vectorizer.fit_transform(df['overview'])
        content_features_df = pd.DataFrame(tfidf_matrix.toarray(), index=df.index)
        return content_features_df
    return None

def get_user_item_interactions(train_df):
    # Create user-item matrix for collaborative filtering (example with ratings)
    user_item_matrix = train_df.pivot_table(index='user_id', columns='movie_id', values='rating').fillna(0)
    return user_item_matrix
# ratings_file = 'ratings.csv'
# movies_file = 'movies.csv'
# ratings_df, movies_df = load_data(ratings_file, movies_file)
# train_df, test_df = preprocess_data(ratings_df, movies_df)
# content_features = get_content_features(train_df.drop_duplicates(subset='movie_id'))
# user_item_matrix = get_user_item_interactions(train_df)
