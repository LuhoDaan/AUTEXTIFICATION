import pandas as pd
from sklearn.model_selection import train_test_split
import os
from typing import Literal, List, Dict
from corpus import Corpus


class Autextication(Corpus):

    identifier: str = 'autextication'

    def __init__(self, split: Literal['train', 'validation', 'test']|None,
                 corpora_path: str, task: str = 'mono'):

        super().__init__(split, corpora_path, task)
        self.corpus_path = os.path.join(self.corpora_path, 'AUTEXTIFICATION/AUTEXTIFICATION')

    def _split_data(self, data) -> List[Dict]:

        # split the src into train, test and validation sets
        train_data, val_data = train_test_split(data, test_size=0.20, random_state=42)
        train_data, test_data = train_test_split(train_data, test_size=0.20, random_state=42)
        if self.split == 'train': return train_data
        if self.split == 'validation': return val_data
        if self.split == 'test': return test_data

    def load_data(self) -> List[Dict]:
        if self.task == 'multi':
            return self._load_multitask_data()
        elif self.task == 'bot':
            return self._load_bot_data()
        else:

            corpus_path = os.path.join(self.corpus_path, 'subtask_1/en/train.tsv')
            text_list = []
            # Read the TSV file into a Pandas dataframe
            df = pd.read_csv(
                corpus_path,
                delimiter='\t')

            for index, row in df.iterrows():
                text = row['text']
                label = row['label']
                # do something with the text and label variables

                text_dict = {
                    'text': text,
                    'origin': label
                }
                text_list.append(text_dict)

            if self.task == 'ppl':
                return text_list
            else:
                return self._split_data(text_list)

    def get_identifier(self) -> str:
        return self.identifier

    def map_bot_to_int(self, bot: str) -> int:
            if bot is not None:
                if bot == 'A':
                    bot = 0
                elif bot == 'B':
                    bot = 1
                elif bot == 'C':
                    bot = 2
                elif bot == 'D':
                    bot = 3
                elif bot == 'E':
                    bot = 4
                elif bot == 'F':
                    bot = 5
                elif bot == 'human-generated':
                    bot = 6
            else:
                bot = -1
            return bot


    def _load_multitask_data(self) -> List[Dict]:

        en1_path = os.path.join(self.corpus_path, 'subtask_1/en/train.tsv')
        en1_list = []
        # Read the TSV file into a Pandas dataframe
        df = pd.read_csv(
            en1_path,
            delimiter='\t')

        for index, row in df.iterrows():
            text = row['text']
            label = row['label']
            if label == 'human':
                bot = 'human-generated'
            else:
                bot = None

            text_dict = {
                'text': text,
                'origin': label,
                'bot': self.map_bot_to_int(bot),
                'language': 'en'

            }
            en1_list.append(text_dict)

        es1_path = os.path.join(self.corpus_path, 'subtask_1/es/train.tsv')
        es1_list = []
        # Read the TSV file into a Pandas dataframe
        df = pd.read_csv(
            es1_path,
            delimiter='\t')

        for index, row in df.iterrows():
            text = row['text']
            label = row['label']
            if label == 'human':
                bot = 'human-generated'
            else:
                bot = None

            text_dict = {
                'text': text,
                'origin': label,
                'bot': self.map_bot_to_int(bot),
                'language': 'es'

            }
            es1_list.append(text_dict)

        en2_path = os.path.join(self.corpus_path, 'subtask_2/en/train.tsv')
        en2_list = []
        # Read the TSV file into a Pandas dataframe
        df = pd.read_csv(
            en2_path,
            delimiter='\t')

        for index, row in df.iterrows():
            text = row['text']
            label = row['label']

            text_dict = {
                'text': text,
                'origin': 'generated',
                'bot': self.map_bot_to_int(label),
                'language': 'en'

            }
            en2_list.append(text_dict)

        es2_path = os.path.join(self.corpus_path, 'subtask_2/es/train.tsv')
        es2_list = []
        # Read the TSV file into a Pandas dataframe
        df = pd.read_csv(
            es2_path,
            delimiter='\t')

        for index, row in df.iterrows():
            text = row['text']
            label = row['label']
            # do something with the text and label variables

            text_dict = {
                'text': text,
                'origin': 'generated',
                'bot': self.map_bot_to_int(label),
                'language': 'es'

            }
            es2_list.append(text_dict)

        engl_list = en1_list + en2_list
        unique_texts = []
        eng_unique_data = []
        for item in engl_list:
            if item['text'] not in unique_texts:
                unique_texts.append(item['text'])
                eng_unique_data.append(item)
            else:
                for i,t in enumerate(eng_unique_data):
                    if t['text'] == item['text']:
                        if t['bot'] < item['bot']:
                            eng_unique_data[i] = item

        esp_list = es1_list + es2_list
        unique_texts = []
        esp_unique_data = []
        for item in esp_list:
            if item['text'] not in unique_texts:
                unique_texts.append(item['text'])
                esp_unique_data.append(item)
            else:
                for i, t in enumerate(esp_unique_data):
                    if t['text'] == item['text']:
                        if t['bot'] < item['bot']:
                            esp_unique_data[i] = item

        full_list = eng_unique_data + esp_unique_data

        return self._split_data(full_list)

    def _load_bot_data(self) -> List[Dict]:

            corpus_path = os.path.join(self.corpus_path, 'subtask_2/en/train.tsv')
            en_text_list = []
            # Read the TSV file into a Pandas dataframe
            df = pd.read_csv(
                corpus_path,
                delimiter='\t')

            for index, row in df.iterrows():
                text = row['text']
                label = row['label']
                # do something with the text and label variables

                text_dict = {
                    'text': text,
                    'bot': label
                }
                en_text_list.append(text_dict)

            corpus_path = os.path.join(self.corpus_path, 'subtask_2/es/train.tsv')
            es_text_list = []
            # Read the TSV file into a Pandas dataframe
            df = pd.read_csv(
                corpus_path,
                delimiter='\t')

            for index, row in df.iterrows():
                text = row['text']
                label = row['label']
                # do something with the text and label variables

                text_dict = {
                    'text': text,
                    'bot': label
                }
                es_text_list.append(text_dict)

            text_list = en_text_list + es_text_list

            return self._split_data(text_list)









