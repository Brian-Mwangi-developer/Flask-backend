import datetime
from vcon import Vcon
from vcon.party import Party
from vcon.dialog import Dialog

# Function to create vCon
def create_vcon(username, most_liked_comment, most_negative_comment, post_url, comment_time_range, likes_count, ai_analysis_positive, ai_analysis_negative, sentiment_analysis):
    
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
        body=most_liked_comment
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
        body=ai_analysis_positive
    )
    vcon.add_dialog(ai_response_liked)

    # Add dialog for the most negative comment
    negative_dialog = Dialog(
        type="text",
        start=start_time,
        parties=[0, 1],  # User is the originator
        originator=0,
        mimetype="text/plain",
        body=most_negative_comment
    )
    vcon.add_dialog(negative_dialog)

    # AI analysis response for the most negative comment
    ai_response_negative = Dialog(
        type="text",
        start=response_time,
        parties=[0, 1],
        originator=1,
        mimetype="text/plain",
        body=ai_analysis_negative
    )
    vcon.add_dialog(ai_response_negative)

    # Add metadata for post URL and comment time range
    vcon.add_tag("post_url", post_url)
    vcon.add_tag("comment_time_range", comment_time_range)

    # Add tags for like counts of both comments
    vcon.add_tag("most_liked_comment_likes", likes_count["liked"])
    vcon.add_tag("most_negative_comment_likes", likes_count["negative"])

    # Add sentiment analysis (positivity, complaints, desires)
    sentiment_analysis_data = {
        "positivity": sentiment_analysis["positivity"],
        "complaints": sentiment_analysis["complaints"],
        "desires": sentiment_analysis["desires"]
    }
    vcon.add_analysis(
        type="sentiment",
        dialog=[0, 1],  # Analyze both comments
        vendor="AI Sentiment Analyzer",
        body=sentiment_analysis_data,
        encoding="none"
    )

    # Add overall verdict and recommendations
    ai_verdict = {
        "overall_verdict": sentiment_analysis["verdict"],
        "recommendations": sentiment_analysis["recommendations"]
    }
    vcon.add_analysis(
        type="verdict",
        dialog=[0, 1],
        vendor="AI Verdict Generator",
        body=ai_verdict,
        encoding="none"
    )

    # Generate a key pair for signing
    private_key, public_key = Vcon.generate_key_pair()

    # Sign the vCon
    vcon.sign(private_key)

    # Verify the signature
    is_valid = vcon.verify(public_key)
    print(f"Signature is valid: {is_valid}")

    # Return the vCon
    return vcon
# Function to simulate data and create a vCon
def simulate_vcon_creation():
    username = "InstagramUser123"
    most_liked_comment = "This product is amazing! Totally changed my life!"
    most_negative_comment = "I'm really disappointed. The quality is terrible."
    post_url = "https://www.instagram.com/p/xyz"
    comment_time_range = "Oct 5, 2024, 2:00 PM - Oct 6, 2024, 4:00 PM"
    likes_count = {"liked": 150, "negative": 10}
    
    ai_analysis_positive = "The comment expresses high satisfaction with the product and is likely to influence other potential customers positively."
    ai_analysis_negative = "The comment highlights dissatisfaction with product quality and may raise concerns among other users."
    
    sentiment_analysis = {
        "positivity": ["Great product", "Changed my life", "Highly recommended"],
        "complaints": ["Poor quality", "Disappointment", "Terrible experience"],
        "desires": ["Better customer support", "Improved quality", "Longer warranty"],
        "verdict": "Mixed reviews with overall positive reception but concerns on product quality.",
        "recommendations": "Focus on improving quality control and addressing customer complaints swiftly."
    }

    # Create vCon
    vcon = create_vcon(username, most_liked_comment, most_negative_comment, post_url, comment_time_range, likes_count, ai_analysis_positive, ai_analysis_negative, sentiment_analysis)
    
    # Print vCon as JSON
    print(vcon.to_json())

if __name__ == "__main__":
    simulate_vcon_creation()
