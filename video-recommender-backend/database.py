# database.py

from flask_pymongo import PyMongo
from gridfs import GridFS

class Database:
    def __init__(self):
        self.mongo = PyMongo()
        self.db = None
        self.grid_fs = None

    def init_app(self, app):
        self.mongo.init_app(app)
        self.db = self.mongo.db
        self.grid_fs = GridFS(self.db)

db = Database()

def init_db(app):
    db.init_app(app)
    return db
