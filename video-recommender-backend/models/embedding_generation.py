import numpy as np
from sklearn.preprocessing import normalize
from database import mongo
from bson.objectid import ObjectId

def generate_embeddings(video_id):
    # Retrieve features from database
    video = mongo.db.videos.find_one({'_id': ObjectId(video_id)})
    features = video['features']

    # Combine features into a single vector
    combined_features = np.concatenate([
        np.array(features['visual']),
        np.array(features['audio']),
        np.array(features['text'])
    ])

    # Generate embedding (e.g., normalize)
    embedding = normalize(combined_features.reshape(1, -1))[0]

    # Store embedding in database
    mongo.db.videos.update_one(
        {'_id': ObjectId(video_id)},
        {'$set': {'embedding': embedding.tolist()}}
    )
