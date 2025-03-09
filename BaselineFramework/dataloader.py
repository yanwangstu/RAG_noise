import json


# load experiment data
class DataLoad:
    def __init__(self, data_paths: list[str]):
        self.data_paths = data_paths
        self.data_contents = []
        self.data_files_samples_num = []
        self.data_files_num = len(data_paths)
        for path in data_paths:
            with open(path, 'r', encoding='utf-8') as file:
                self.data_contents.append(json.load(file))
                self.data_files_samples_num.append(len(self.data_contents[-1]))
        # the inner dict is samples in each data_file
        self.data_contents: list[dict]

    def get_data(self,
                 data_files_id: int,
                 data_files_samples_id: int
                 ) -> dict:
        return self.data_contents[data_files_id][data_files_samples_id]
