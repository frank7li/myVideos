# models/ranking_model.py
import lightgbm as lgb

class RankingModel:
    def __init__(self):
        self.model = lgb.Booster(model_file='ranking_model.txt')  # Load your trained model

    def predict(self, features):
        return self.model.predict(features)
