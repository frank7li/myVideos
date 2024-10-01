from flask import Flask
from flask_bcrypt import Bcrypt
from flask_jwt_extended import JWTManager
from flask_cors import CORS

app = Flask(__name__)
bcrypt = Bcrypt()
jwt = JWTManager()
cors = CORS()

def init_app(config_class):
    app.config.from_object(config_class)
    bcrypt.init_app(app)
    jwt.init_app(app)
    cors.init_app(app, resources={r"/api/*": {"origins": "*"}})
    return app