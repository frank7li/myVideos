# app.py 

from flask import Flask, request, jsonify
from flask_bcrypt import Bcrypt
from flask_jwt_extended import (
    JWTManager, create_access_token,
    jwt_required, get_jwt_identity
)
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import numpy as np
from datetime import datetime
from bson.objectid import ObjectId
from tasks import process_video_task
from models.embedding_generation import generate_user_embedding  # Ensure this is defined
from database import mongo
import boto3
from flask_pymongo import PyMongo
from gridfs import GridFS
from config import Config
from database import init_db, mongo, grid_fs







app = Flask(__name__)
CORS(app)
app.config.from_object(Config)

# Configuration
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MONGO_URI'] = 'mongodb+srv://<username>:<password>@cluster0.mongodb.net/mydatabase?retryWrites=true&w=majority'
app.config['S3_BUCKET'] = 'your-s3-bucket-name'
app.config['JWT_SECRET_KEY'] = 'your-secret-key'  # Change this!

bcrypt = Bcrypt(app)
jwt = JWTManager(app)

init_db(app)

# Initialize MongoDB
mongo = PyMongo(app)
db = mongo.db
grid_fs = GridFS(db)

# Initialize S3 client securely
s3_client = boto3.client(
    's3',
    aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY'),
    region_name='YOUR_REGION'
)

# Routes
@app.route('/api/upload', methods=['POST'])
@jwt_required()
def upload_video():
    current_user = get_jwt_identity()
    user = mongo.db.users.find_one({'email': current_user})

    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    video = request.files['video']
    filename = secure_filename(video.filename)
    video.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

    # Upload to S3
    s3_client.upload_file(
        Filename=os.path.join(app.config['UPLOAD_FOLDER'], filename),
        Bucket=app.config['S3_BUCKET'],
        Key=filename
    )

    # Store video metadata in MongoDB
    video_doc = {
        'filename': filename,
        'user_id': user['_id'],
        's3_key': filename,
        'features': None,
        'created_at': datetime.utcnow()
    }
    video_id = mongo.db.videos.insert_one(video_doc).inserted_id

    # Trigger video processing task
    process_video_task.delay(filename, str(video_id), str(user['_id']))

    return jsonify({'message': 'Video uploaded and processing started'}), 200

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
    recommendations = get_recommendations_for_user(user['_id'])

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

if __name__ == '__main__':
    app.run(debug=True)
