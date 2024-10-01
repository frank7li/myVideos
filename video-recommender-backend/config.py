# config.py

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    SECRET_KEY = os.environ['SECRET_KEY']
    MONGO_URI = os.environ['MONGO_URI']
    S3_BUCKET = os.environ['S3_BUCKET']
    AWS_ACCESS_KEY_ID = os.environ['AWS_ACCESS_KEY_ID']
    AWS_SECRET_ACCESS_KEY = os.environ['AWS_SECRET_ACCESS_KEY']
    AWS_REGION = os.environ['AWS_REGION']
    UPLOAD_FOLDER = os.environ.get('UPLOAD_FOLDER', 'uploads')
    JWT_SECRET_KEY = os.environ['JWT_SECRET_KEY']
    REDIS_URL = os.environ.get('REDIS_URL', 'redis://localhost:6379/0')

    @classmethod
    def validate(cls):
        required_vars = [
            'SECRET_KEY', 'MONGO_URI', 'S3_BUCKET', 'AWS_ACCESS_KEY_ID',
            'AWS_SECRET_ACCESS_KEY', 'AWS_REGION', 'JWT_SECRET_KEY'
        ]
        missing = [var for var in required_vars if not os.environ.get(var)]
        if missing:
            raise ValueError(f"Missing required environment variables: {', '.join(missing)}")
