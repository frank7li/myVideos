from extensions import app, jwt, bcrypt
from flask import request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity, create_access_token
from database import mongo
from tasks import process_video_task
from werkzeug.utils import secure_filename
import os
import random
import string
import numpy as np
from datetime import datetime
from bson.objectid import ObjectId
from models.embedding_generation import generate_user_embedding
from models.candidate_ranking import get_recommendations_for_user

# Function to generate a random 10-character string
def generate_unique_id():
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=10))

@app.route('/api/upload', methods=['POST'])
@jwt_required()
def upload_video():
    current_user = get_jwt_identity()
    user = mongo.db.users.find_one({'email': current_user})

    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400

    video = request.files['video']
    original_filename = secure_filename(video.filename)

    unique_id = generate_unique_id()
    while mongo.db.videos.find_one({'video_id': unique_id}):
        unique_id = generate_unique_id()

    filename = f"{unique_id}_{original_filename}"

    video.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

    try:
        app.s3_client.upload_file(
            Filename=os.path.join(app.config['UPLOAD_FOLDER'], filename),
            Bucket=app.config['S3_BUCKET'],
            Key=filename
        )

        video_doc = {
            'video_id': unique_id,
            'filename': filename,
            'user_id': user['_id'],
            's3_key': filename,
            'features': None,
            'created_at': datetime.now(datetime.timezone.utc)
        }

        video_id = mongo.db.videos.insert_one(video_doc).inserted_id

        process_video_task.delay(filename, str(video_id), str(user['_id']), 
                            app.config['UPLOAD_FOLDER'], app.config['S3_BUCKET'])
    except Exception as e:
        return jsonify({'error': str(e)}), 500


    os.remove(os.path.join(app.config['UPLOAD_FOLDER'], filename))

    return jsonify({'message': 'Video uploaded and processing started', 'video_id': unique_id}), 200

@app.route('/api/recommendations', methods=['GET'])
@jwt_required()
def get_recommendations():
    current_user = get_jwt_identity()
    user = mongo.db.users.find_one({'email': current_user})

    # Generate or retrieve user embedding
    if 'embedding' not in user:
        # Generate user embedding based on interactions
        user_embedding = generate_user_embedding(user['_id'])
        mongo.db.users.update_one(
            {'_id': user['_id']},
            {'$set': {'embedding': user_embedding.tolist()}}
        )
    else:
        user_embedding = np.array(user['embedding'])

    # Get recommendations
    recommendations = get_recommendations_for_user(str(user['_id']), s3_bucket=app.config['S3_BUCKET'])

    return jsonify({'videos': recommendations}), 200

@app.route('/api/register', methods=['POST'])
def register():
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')

    if mongo.db.users.find_one({'email': email}):
        return jsonify({'error': 'User already exists'}), 409

    password_hash = bcrypt.generate_password_hash(password).decode('utf-8')

    mongo.db.users.insert_one({
        'email': email,
        'password_hash': password_hash
    })

    return jsonify({'message': 'User registered successfully'}), 201

@app.route('/api/login', methods=['POST'])
def login():
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')

    user = mongo.db.users.find_one({'email': email})
    if not user or not bcrypt.check_password_hash(user['password_hash'], password):
        return jsonify({'error': 'Invalid email or password'}), 401

    access_token = create_access_token(identity=user['email'])
    return jsonify({'access_token': access_token}), 200