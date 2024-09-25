# database.py

from flask_pymongo import PyMongo
from gridfs import GridFS

mongo = PyMongo()
db = mongo.db
grid_fs = None

def init_db(app):
    mongo.init_app(app)
    global db, grid_fs
    db = mongo.db
    grid_fs = GridFS(db)
