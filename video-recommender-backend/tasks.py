# tasks.py

from celery_app import celery
from models.feature_extraction import extract_features
from models.embedding_generation import generate_embeddings
from database import db
import os
from bson.objectid import ObjectId

@celery.task
def process_video_task(filename, video_id, user_id):
    from app import app  # Import within the function to avoid circular import
    # Path to the uploaded video
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    # Step 1: Extract Features
    features = extract_features(video_path)

    # Step 2: Store Features in Database
    db.videos.update_one(
        {'_id': ObjectId(video_id)},
        {'$set': {'features': features}}
    )

    # Step 3: Generate Embeddings
    generate_embeddings(video_id)

    # Optional: Update Recommendations
    # update_recommendations_for_user(user_id)

    print(f"Video {video_id} processed successfully.")
