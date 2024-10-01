import numpy as np
from sklearn.preprocessing import normalize
from database import mongo
from bson.objectid import ObjectId
from sklearn.decomposition import PCA

def generate_embeddings(video_id):
    # Retrieve features from database
    video = mongo.db.videos.find_one({'_id': ObjectId(video_id)})
    features = video['features']

    # Convert torch tensor to numpy array
    features_np = features.cpu().numpy()

    # Apply PCA to reduce dimensionality from 512 to 128
    pca = PCA(n_components=128)
    reduced_features = pca.fit_transform(features_np.reshape(1, -1))

    # Normalize the reduced features
    embedding = normalize(reduced_features)[0]

    # Store embedding in database
    mongo.db.videos.update_one(
        {'_id': ObjectId(video_id)},
        {'$set': {'embedding': embedding.tolist()}}
    )


def generate_user_embedding(user_id):
    # Retrieve user's interactions from database
    #placeholder
    return np.random.rand(50)
