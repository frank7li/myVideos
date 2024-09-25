# config.py

import os

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'your-secret-key'
    MONGO_URI = os.environ.get('MONGO_URI') or 'mongodb+srv://<username>:<password>@cluster0.mongodb.net/mydatabase?retryWrites=true&w=majority'
    S3_BUCKET = os.environ.get('S3_BUCKET') or 'your-s3-bucket-name'
    AWS_ACCESS_KEY_ID = os.environ.get('AWS_ACCESS_KEY_ID')
    AWS_SECRET_ACCESS_KEY = os.environ.get('AWS_SECRET_ACCESS_KEY')
    AWS_REGION = os.environ.get('AWS_REGION') or 'your-region'
    UPLOAD_FOLDER = 'uploads'
    JWT_SECRET_KEY = os.environ.get('JWT_SECRET_KEY') or 'your-jwt-secret-key'
    REDIS_URL = os.environ.get('REDIS_URL') or 'redis://localhost:6379/0'
