import pandas as pd
from openai import OpenAI
import os
import json
import numpy as np
import pickle
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

# Convert the scraped data into a Pandas DataFrame
df = pd.json_normalize(comments_data["data"])

# Function to translate text to English
def translate_to_english(text):
    prompt = f"You are a translator. Only provide the translation to English. Text: {text}"
    if text.strip():  # Avoid translating empty text
        try:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=20,
                temperature=0.5
            )
            text = response.choices[0].message.content.strip()
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

# Function to save embeddings using pickle
def save_embeddings(embeddings, filename='comments_embeddings.pkl'):
    try:
        with open(filename, 'wb') as f:
            pickle.dump(embeddings, f)
        print(f"Embeddings saved to {filename}")
    except Exception as e:
        print(f"Error saving embeddings: {e}")

# Function to analyze sentiment
def analyze_sentiment(text):
    analysis = TextBlob(text)
    return analysis.sentiment.polarity  # Returns a value between -1 and 1

# Function to analyze comment desires, complaints, or positivity using OpenAI
def analyze_comment_summary(comments):
    text = "\n".join(comments)
    prompt = f"Summarize the following comments into desires, complaints, and positivity in a numbered format:\n{text}"
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150,
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error generating summary: {e}")
        return "No summary available."

# Function to extract themes based on embeddings, likes, and sentiment
def extract_themes(df):
    # Translate comments to English
    df['translated_text'] = df['text'].apply(lambda x: translate_to_english(x) if x else "")
    
    # Get embeddings for the translated comments
    embeddings = get_embeddings(df['translated_text'].tolist())
    
    # Save embeddings to a pickle file
    save_embeddings(embeddings)
    
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
            similar_comments = np.where(similarity_matrix[i] > 0.75)[0]  # Adjust threshold as needed
            similar_comments = [idx for idx in similar_comments if idx not in visited]
            
            # Mark these comments as visited
            visited.update(similar_comments)
            
            # Collect similar comments, sorted by likes and sentiment
            if similar_comments:
                similar_df = df.iloc[similar_comments].copy()
                similar_df['sentiment'] = similar_df['translated_text'].apply(analyze_sentiment)
                
                # Sort by likes to get the most liked comment
                most_liked = similar_df.sort_values(by='likesCount', ascending=False).iloc[0]
                
                # Sort by sentiment to get the most negative comment
                most_negative = similar_df.sort_values(by='sentiment').iloc[0]
                
                themes.append({
                    'most_liked': {
                        'comment': most_liked['translated_text'],
                        'likes': most_liked['likesCount'],
                        'sentiment': most_liked['sentiment'],
                    },
                    'most_negative': {
                        'comment': most_negative['translated_text'],
                        'likes': most_negative['likesCount'],
                        'sentiment': most_negative['sentiment'],
                    }
                })
                
    
    # Analyze comment summary using OpenAI
    all_comments = df['translated_text'].tolist()
    summary = analyze_comment_summary(all_comments)
    
    return themes, summary

# Call the function and print themes and summary
themes, summary = extract_themes(df)

print("Extracted Themes:")
for theme in themes:
    print(f"Most Liked Comment: {theme['most_liked']['comment']} | Likes: {theme['most_liked']['likes']} | Sentiment: {theme['most_liked']['sentiment']:.2f}")
    print(f"Most Negative Comment: {theme['most_negative']['comment']} | Likes: {theme['most_negative']['likes']} | Sentiment: {theme['most_negative']['sentiment']:.2f}")
    
print("\nComment Summary:")
print(summary)
