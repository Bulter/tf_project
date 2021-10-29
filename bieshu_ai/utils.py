import os
import json
import pickle


DATA_FILE = r'/home/bulter/tf_project/data'


def lazy_property(func):
    attr_name = f'_lazy_{func.__name__}'

    @property
    def _lazy_proprety(self):
        if not hasattr(self, attr_name):
            setattr(self, attr_name, func(self))
        return getattr(self, attr_name)
    
    return _lazy_proprety


class DataIterator:
    def __init__(self, data, batch_size):
        self.data = data
        self.batch_size = batch_size

        self.start = 0
        self.end = len(data)

    def __iter__(self):
        return self
    
    def __next__(self):
        start = self.start
        if start < self.end:
            self.start += self.batch_size
            return self.data[start:self.start]
        else:
            raise StopIteration


class DataLoaderBase:
    def __init__(self) -> None:
        self.id2label = self._get_by_json('id2label.json')
        self.label2id = self._get_by_json('label2id.json')

        self.id2word = self._get_by_pkl('id2word.pkl')
        self.word2id = self._get_by_pkl('word2id.pkl')

        self.data = self._get_by_pkl('data.pkl')
        
    def _get_by_json(self, file_name):
        with open(os.path.join(DATA_FILE, file_name), 'r', encoding='utf-8') as fp:
            data_dict = json.loads(fp.read())
        return data_dict
    
    def _get_by_pkl(self, file_name):
        with open(os.path.join(DATA_FILE, file_name), 'rb') as fp:
            data_dict = pickle.load(fp)
        return data_dict

    def get_data_iterator(self, data, batch_size):
        return DataIterator(data, batch_size)


