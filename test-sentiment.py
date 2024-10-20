import pandas as pd
from openai import OpenAI
import os
import json
import numpy as np
from textblob import TextBlob
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity

# Load environment variables
load_dotenv()

client = OpenAI()
client.api_key = os.getenv("OPENAI_API_KEY")

# Load comments from JSON file
with open('dummy-com.json', 'r', encoding='utf-8') as file:
    comments_data = json.load(file)

# Convert the scraped Data into a Pandas DataFrame
df = pd.json_normalize(comments_data["data"])

# Function to translate text to English
def translate_to_english(text):
    prompt =f"You are a translator. Only provide the translation to english. Do not generate any additional text or explanations. Only respond with the translation. Text: {text}"
    if text.strip():  # Avoid translating empty text
        try:
            response = client.chat.completions.create(**{
            "model": "gpt-4",
            "messages": [
                {"role": "user", "content": prompt}
             ],
             "max_tokens": 20,
            "temperature": 0.5
            })
            text= response.choices[0].message.content
            print(f"Translation: {text}")
            return text
        except Exception as e:
            print(f"Translation error: {e}")
            return text  # Return original text if translation fails
    return text

# Function to get embeddings from OpenAI
def get_embeddings(texts):
    embeddings = []
    for text in texts:
        try:
            response = client.embeddings.create(
                model="text-embedding-ada-002",
                input=text
            )
            embeddings.append(response.data[0].embedding)
        except Exception as e:
            print(f"Embedding error: {e}")
            embeddings.append([0] * 1536)  # Append a zero vector in case of failure
    return embeddings

# Function to analyze sentiment
def analyze_sentiment(text):
    analysis = TextBlob(text)
    return analysis.sentiment.polarity  # Returns a value between -1 and 1

# Function to extract themes based on embeddings and sentiment
def extract_themes(df):
    # Translate comments to English
    df['translated_text'] = df['text'].apply(lambda x: translate_to_english(x) if x else "")
    
    # Get embeddings for the translated comments
    embeddings = get_embeddings(df['translated_text'].tolist())
    
    # Convert embeddings to a NumPy array
    embeddings_matrix = np.array(embeddings)
    
    # Calculate cosine similarity between comment embeddings
    similarity_matrix = cosine_similarity(embeddings_matrix)
    np.fill_diagonal(similarity_matrix, 0)  # Ignore self-similarity
    
    themes = []
    visited = set()  # To avoid processing the same comment multiple times
    
    for i in range(similarity_matrix.shape[0]):
        if i not in visited:
            # Find comments with high similarity
            similar_comments = np.where(similarity_matrix[i] > 0.75)[0]  
            similar_comments = [idx for idx in similar_comments if idx not in visited]
            
            # Mark these comments as visited
            visited.update(similar_comments)
            
            # Collect similar comments, sorted by likes and sentiment
            if similar_comments:
                similar_df = df.iloc[similar_comments].copy()
                similar_df['sentiment'] = similar_df['translated_text'].apply(analyze_sentiment)
                similar_df = similar_df.sort_values(by=['likesCount', 'sentiment'], ascending=False)
                
                # Take the most liked comment as the representative of this theme
                top_comment = similar_df.iloc[0]
                themes.append({
                    'comment': top_comment['translated_text'],
                    'likes': top_comment['likesCount'],
                    'sentiment': top_comment['sentiment'],
                })
    
    return themes

# Call the function and print themes
themes = extract_themes(df)

print("Extracted Themes:")
for theme in themes:
    print(f"Comment: {theme['comment']} | Likes: {theme['likes']} | Sentiment: {theme['sentiment']:.2f}")
