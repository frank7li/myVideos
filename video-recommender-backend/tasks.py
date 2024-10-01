# tasks.py

from celery_app import celery
from models.feature_extraction import extract_features
from models.embedding_generation import generate_embeddings
from database import db
import os
from bson.objectid import ObjectId

@celery.task
def process_video_task(filename, video_id, user_id, upload_folder, s3_bucket):
    # Path to the uploaded video
    video_path = os.path.join(upload_folder, filename)

    # Extract Features
    features = extract_features(video_path)

    # Store Features in Database
    db.videos.update_one(
        {'_id': ObjectId(video_id)},
        {'$set': {'features': features}}
    )

    # Generate Embeddings
    generate_embeddings(video_id)

    # Optional: Update Recommendations
    # update_recommendations_for_user(user_id)

    print(f"Video {video_id} processed successfully.")
