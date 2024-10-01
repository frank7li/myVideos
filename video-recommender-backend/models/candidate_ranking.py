from database import mongo
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from bson.objectid import ObjectId

def get_recommendations_for_user(user_id, top_n=10, s3_bucket=None):
    # Retrieve user embedding
    user = mongo.db.users.find_one({'_id': ObjectId(user_id)})
    user_embedding = np.array(user.get('embedding'))

    # Retrieve all video embeddings
    videos = list(mongo.db.videos.find({'embedding': {'$exists': True}}))
    video_embeddings = np.array([video['embedding'] for video in videos])

    # Compute similarity scores
    similarities = cosine_similarity([user_embedding], video_embeddings)[0]

    # Get top N videos
    top_indices = similarities.argsort()[-top_n:][::-1]
    recommended_videos = [videos[i] for i in top_indices]

    # Return video information
    recommendations = []
    for video in recommended_videos:
        recommendations.append({
            'id': str(video['_id']),
            'url': f"https://{s3_bucket}.s3.amazonaws.com/{video['s3_key']}"
        })
    return recommendations
