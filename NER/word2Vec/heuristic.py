import logging
from datetime import datetime

class HeuristicManager:
    def __init__(self, model, log_file="heuristic_log.txt", min_similarity_threshold=0.5, min_new_data_len=50):
        self.model = model
        self.min_similarity_threshold = min_similarity_threshold
        self.min_new_data_len = min_new_data_len
        self.log_file = log_file
        logging.basicConfig(filename=self.log_file, level=logging.INFO)

    def log(self, message):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logging.info(f"[{timestamp}] {message}")
        print(f"[{timestamp}] {message}")

    def check_similarity(self, test_terms):
        triggers = []
        for term in test_terms:
            try:
                sim = self.model.wv.most_similar(term)[0][1]
                if sim < self.min_similarity_threshold:
                    triggers.append(f"Low similarity for '{term}': {sim}")
            except KeyError:
                triggers.append(f"'{term}' not in vocabulary")
        return triggers

    def check_metadata(self, metadata):
        triggers = []
        if any(keyword in str(metadata).lower() for keyword in ["haplogroup b", "eastasia", "asian"]):
            triggers.append("Detected new haplogroup or regional bias: 'Asian' or 'B'")
        return triggers

    def check_new_data_volume(self, new_data):
        if len(new_data) < self.min_new_data_len:
            return ["Not enough new data to justify retraining"]
        return []

    def should_retrain(self, test_terms, new_data, metadata):
        triggers = []
        triggers += self.check_similarity(test_terms)
        triggers += self.check_metadata(metadata)
        triggers += self.check_new_data_volume(new_data)

        if triggers:
            self.log("Retraining triggered due to:")
            for trigger in triggers:
                self.log(f" - {trigger}")
            return True
        else:
            self.log("No retraining needed.")
            return False
