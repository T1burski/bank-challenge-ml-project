import json

class DataPreprocessor:
    def __init__(self, data):
        self.raw_data = data
        self.processed_data = None

    def select_columns(self, env="prod"):
        features_json = json.load(open('rfe_results.json'))

        if env == "train":
            selected_columns = features_json["100"] + ['target']
            self.processed_data.drop_duplicates(inplace=True)
        elif env == "prod":
            selected_columns = features_json["100"]
        self.processed_data = self.raw_data[selected_columns]

        return self.processed_data