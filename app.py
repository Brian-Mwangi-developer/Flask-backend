from flask import Flask, request, jsonify

import pandas as pd
from openai import OpenAI
from flask_cors import CORS
import os
import json
import numpy as np
import pickle
from textblob import TextBlob
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from vcon import Vcon
from vcon.party import Party
from vcon.dialog import Dialog
import datetime
from mongoconnect import save_vcon
from mongoconnect import collection 

# Load environment variables
load_dotenv()
app = Flask(__name__)
CORS(app)
client = OpenAI()
client.api_key = os.getenv("OPENAI_API_KEY")

# Load comments from JSON file
with open('dummy-com.json', 'r', encoding='utf-8') as file:
    comments_data = json.load(file)

analyzer = SentimentIntensityAnalyzer()
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

def categorize_sentiment(text):
    sentiment_scores = analyzer.polarity_scores(text)
    if sentiment_scores["compound"] >= 0.5:
        return "Positive"
    elif sentiment_scores["compound"] <= -0.5:
        return "Negative"
    else:
        return "Neutral"
    
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
    
    # Get the most liked and negative comments
    most_liked_comment = None
    most_negative_comment = None
    highest_likes = -1
    lowest_sentiment = 1
    
    positive_count = 0
    negative_count = 0
    neutral_count = 0
    
    for i in range(similarity_matrix.shape[0]):
        # Get sentiment for each comment
        sentiment_category = categorize_sentiment(df['translated_text'].iloc[i])
        likes = df['likesCount'].iloc[i]
        # Count the Sentiment category
        if sentiment_category == 'positive':
            positive_count +=1
        elif sentiment_category == 'negative':
            negative_count +=1
        else:
            neutral_count += 1
        # Check for the most liked comment
        if likes > highest_likes:
            highest_likes = likes
            most_liked_comment = {
                'comment': df['translated_text'].iloc[i],
                'likes': likes,
                'sentiment': sentiment_category,
            }

        # Check for the most negative comment
        sentiment_score = analyze_sentiment(df['translated_text'].iloc[i])  # Use TextBlob for score
        if sentiment_score< lowest_sentiment:
            lowest_sentiment = sentiment_score
            most_negative_comment = {
                'comment': df['translated_text'].iloc[i],
                'likes': likes,
                'sentiment': sentiment_category,
            }

    # Save themes
    themes.append({
        'most_liked': most_liked_comment,
        'most_negative': most_negative_comment
    })

    # Analyze comment summary using OpenAI
    all_comments = df['translated_text'].tolist()
    summary = analyze_comment_summary(all_comments)
    
    return themes, summary, positive_count, negative_count, neutral_count

# Function to create and save a vCon
def create_vcon(username, most_liked_comment, most_negative_comment, summary, positive_count, negative_count, neutral_count, post_url):
    # Create a new vCon object
    
    vcon = Vcon.build_new()

    # Add parties
    user_party = Party(tel="", name=username, role="Instagram comment")
    ai_party = Party(tel="", name="AI Analysis", role="Sentiment Analysis")
    vcon.add_party(user_party)
    vcon.add_party(ai_party)

    # Add dialog for the most liked comment
    start_time = datetime.datetime.now().isoformat()
    liked_dialog = Dialog(
        type="text",
        start=start_time,
        parties=[0, 1],  # User is the originator
        originator=0,  # User who commented
        mimetype="text/plain",
        body=most_liked_comment['comment']
    )
    vcon.add_dialog(liked_dialog)

    # AI analysis response for the most liked comment
    response_time = (datetime.datetime.now() + datetime.timedelta(minutes=1)).isoformat()
    ai_response_liked = Dialog(
        type="text",
        start=response_time,
        parties=[0, 1],  # AI is responding to the comment
        originator=1,  # AI as the originator
        mimetype="text/plain",
        body=f"Sentiment: {most_liked_comment['sentiment']} | Likes: {most_liked_comment['likes']}"
    )
    vcon.add_dialog(ai_response_liked)

    # Add dialog for the most negative comment
    negative_dialog = Dialog(
        type="text",
        start=start_time,
        parties=[0, 1],  # User is the originator
        originator=0,
        mimetype="text/plain",
        body=most_negative_comment['comment']
    )
    vcon.add_dialog(negative_dialog)

    # AI analysis response for the most negative comment
    ai_response_negative = Dialog(
        type="text",
        start=response_time,
        parties=[0, 1],
        originator=1,
        mimetype="text/plain",
        body=f"Sentiment: {most_negative_comment['sentiment']} | Likes: {most_negative_comment['likes']}"
    )
    vcon.add_dialog(ai_response_negative)

    # # Add summary of comments
    # vcon.add_tag("comment_summary", summary)
    vcon.add_analysis(
        type="sentiment",
        dialog=[0, 1],  # Indices of the dialogs analyzed
        vendor="SentimentAnalyzer",
        body=summary,
        encoding="none"
    )
    vcon.add_tag("postUrl", post_url)
    
    vcon.add_tag("positive_comments_count", positive_count)
    vcon.add_tag("negative_comments_count", negative_count)
    vcon.add_tag("neutral_comments_count", neutral_count)
    # Generate a key pair for signing
    private_key, public_key = Vcon.generate_key_pair()

    # Sign the vCon
    vcon.sign(private_key)

    # Verify the signature
    is_valid = vcon.verify(public_key)
    print(f"Signature is valid: {is_valid}")
    
    with open('first-vcon.json', 'w') as vcon_file:
        json.dump(vcon.to_dict(), vcon_file, indent=4)
    print("vCon saved to first-vcon.json")

    # Return the vCon
    return vcon

def extract_sentiment_data(body, positivity_list, complaints_list, desires_list):
    # Initialize lists to store extracted sentiments
    desires_found = False
    complaints_found = False
    positivity_found = False

    # Split the body into lines for analysis
    for line in body.split("\n"):
        # Check for "Desires" section
        if "Desires:" in line:
            desires_found = True
            continue
        elif desires_found and line.strip() == "":
            desires_found = False  # Stop if we hit an empty line
            continue
        
        # Check for "Complaints" section
        if "Complaints:" in line:
            complaints_found = True
            continue
        elif complaints_found and line.strip() == "":
            complaints_found = False  # Stop if we hit an empty line
            continue
        
        # Check for "Positivity" section
        if "Positivity:" in line:
            positivity_found =True
            continue
        elif positivity_found and line.strip() == "":
            positivity_found = False  # Stop if we hit an empty line
            continue

        # Extract desires and complaints
        if desires_found:
            desires_list.append(line.strip())
        elif complaints_found:
            complaints_list.append(line.strip())
        elif positivity_found:
            positivity_list.append(line.strip())
# Function to count sentiment
def count_sentiments(vcons):
    total_positive = 0
    total_negative = 0
    total_neutral = 0

    positivity_list = []
    complaints_list = []
    desires_list = []

    for vcon in vcons:
        # Check if 'attachments' field exists and process the sentiment counts
        if "attachments" in vcon:
            for attachment in vcon["attachments"]:
                if attachment["type"] == "tags":
                    for tag in attachment["body"]:
                        if "positive_comments_count" in tag:
                            total_positive += int(tag.split(":")[1])
                        if "negative_comments_count" in tag:
                            total_negative += int(tag.split(":")[1])
                        if "neutral_comments_count" in tag:
                            total_neutral += int(tag.split(":")[1])

        # Check if 'analysis' field exists and extract sentiment insights
        if "analysis" in vcon:
            analysis = vcon["analysis"]
            
            # If 'analysis' is a list, iterate through it
            if isinstance(analysis, list):
                for item in analysis:
                    body = item.get("body", "")
                    extract_sentiment_data(body, positivity_list, complaints_list, desires_list)
            
            # If 'analysis' is a dictionary, handle directly
            elif isinstance(analysis, dict):
                body = analysis.get("body", "")
                extract_sentiment_data(body, positivity_list, complaints_list, desires_list)

    return {
        "total_positive": total_positive,
        "total_negative": total_negative,
        "total_neutral": total_neutral,
        "positivity": positivity_list,
        "complaints": complaints_list,
        "desires": desires_list
    }  

def load_embeddings(filename='comments_embeddings.pkl'):
    try:
        with open(filename, 'rb') as f:
            embeddings = pickle.load(f)
        print(f"Embeddings loaded from {filename}")
        return np.array(embeddings)
    except Exception as e:
        print(f"Error loading embeddings: {e}")
        return None
    
def get_sentiment(text):
    analyzer = SentimentIntensityAnalyzer()
    sentiment_scores = analyzer.polarity_scores(text)
    if sentiment_scores["compound"] >= 0.5:
        return "Positive"
    elif sentiment_scores["compound"] <= -0.5:
        return "Negative"
    else:
        return "Neutral"
    
def similarity_search(query, stored_embeddings):
    # Generate the query embedding
    query_embedding = get_embeddings(query)

    # Ensure stored embeddings are 2D
    if stored_embeddings.ndim == 3:
        stored_embeddings = stored_embeddings.reshape(len(stored_embeddings), -1)

    # Compute cosine similarity
    similarities = cosine_similarity(query_embedding, stored_embeddings)
    return similarities.flatten()  # Flatten for easier access
@app.route("/search", methods=["POST"])
def search():
    data = request.json
    query = data.get('query')

    # Load the stored embeddings once and reuse them
    stored_embeddings = load_embeddings()  # Load embeddings only once

    # Perform similarity search
    similarities = similarity_search(query, stored_embeddings)

    # Find the top results (for example, top 5)
    top_indices = similarities.argsort()[-5:][::-1]

    # Convert results to standard Python types for JSON serialization
    results = [{"index": int(idx), "similarity": float(similarities[idx].item())} for idx in top_indices]

    return jsonify(results)
@app.route("/retrive-sentiment", methods=["GET"])
def retrive_sentiment():
    try:
        vcons =list(collection.find({}))
        sentiment_data = count_sentiments(vcons)
        return jsonify({
            "total_positive": sentiment_data["total_positive"],
            "total_negative": sentiment_data["total_negative"],
            "total_neutral": sentiment_data["total_neutral"],
            "positivity": sentiment_data["positivity"],
            "complaints": sentiment_data["complaints"],
            "desires": sentiment_data["desires"] 
        }),200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    

@app.route("/process-comments", methods=["POST"])
def process_comments():
    # Get the comments from the request
    try:
        comments_data = request.get_json(force=True)
        if not comments_data:
            return jsonify({"error": "No comments provided"}), 400

        # Convert the scraped data into a Pandas DataFrame
        df = pd.json_normalize(comments_data["data"])
        
         # Get the post URL from the DataFrame
        post_url = df['postUrl'].iloc[0] if 'postUrl' in df.columns else "Unknown"
        # Call the function and print themes and summary
        themes, summary, positive_count, negative_count, neutral_count = extract_themes(df)

        # Assuming the username is stored in 'ownerUsername'
        username = df['ownerUsername'].iloc[0] if 'ownerUsername' in df.columns else "Unknown"

        vcon = None
        # Create and save the vCon
        if themes:
            vcon = create_vcon(username, themes[0]['most_liked'], themes[0]['most_negative'], summary, positive_count, negative_count, neutral_count,post_url)
        # Save vCon to MongoDB
        if vcon:
            save_vcon(vcon.to_dict())

        return jsonify({"success": True}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
if __name__ == "__main__":
    app.run(debug=True)