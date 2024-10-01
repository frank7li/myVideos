import numpy as np
from sklearn.preprocessing import normalize
from database import mongo
from bson.objectid import ObjectId
from sklearn.decomposition import PCA

def generate_embeddings(video_id):
    # Retrieve features from database
    video = mongo.db.videos.find_one({'_id': ObjectId(video_id)})
    features = video['features']

    # Apply PCA to reduce dimensionality
    pca_visual = PCA(n_components=50).fit_transform(np.array(features['visual']))
    pca_audio = PCA(n_components=50).fit_transform(np.array(features['audio']))
    pca_text = PCA(n_components=50).fit_transform(np.array(features['text']))

    # Combine features into a single vector
    combined_features = np.concatenate([pca_visual, pca_audio, pca_text])

    # Generate embedding (e.g., normalize)
    embedding = normalize(combined_features.reshape(1, -1))[0]

    # Store embedding in database
    mongo.db.videos.update_one(
        {'_id': ObjectId(video_id)},
        {'$set': {'embedding': embedding.tolist()}}
    )


def generate_user_embedding(user_id):
    # Retrieve user's interactions from database
    #placeholder
    return np.random.rand(50)
