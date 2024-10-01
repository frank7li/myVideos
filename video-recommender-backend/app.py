# app.py

from extensions import init_app
from config import Config
from database import init_db
import boto3

Config.validate()
app = init_app(Config)
db = init_db(app)

# Initialize S3 client securely
app.s3_client = boto3.client(
    's3',
    aws_access_key_id=app.config['AWS_ACCESS_KEY_ID'],
    aws_secret_access_key=app.config['AWS_SECRET_ACCESS_KEY'],
    region_name=app.config['AWS_REGION']
)

import routes

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
