# models/ranking_model.py
import lightgbm as lgb

class RankingModel:
    def __init__(self):
        self.model = lgb.Booster(model_file='ranking_model.txt')  # A pre-trained model, havn't collected enough data for training yet

    def predict(self, features):
        return self.model.predict(features)
