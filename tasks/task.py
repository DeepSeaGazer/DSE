import json


class MultipleChoiceTask:
    def __init__(self, file_path):
        self.file_path = file_path
        self._index = 0
        self._data = self._load_json()

    def __len__(self):
        return len(self._data)

    def _load_json(self):
        try:
            with open(self.file_path, 'r') as file:
                data = json.load(file)
                if not isinstance(data, list):
                    raise ValueError("JSON content is not a list")
                return data
        except FileNotFoundError:
            raise FileNotFoundError(f"The file {self.file_path} does not exist")
        except json.JSONDecodeError:
            raise ValueError("The file does not contain valid JSON")

    def __iter__(self):
        # Reset the index each time iter is called
        self._index = 0
        return self

    def __next__(self):
        try:
            item = self._data[self._index]
        except IndexError:
            # If we reach the end of the list, we raise StopIteration
            raise StopIteration
        self._index += 1
        return item
