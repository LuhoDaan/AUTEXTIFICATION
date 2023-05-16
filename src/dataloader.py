import pandas as pd
from sklearn.model_selection import train_test_split
import bz2
import pickle
import os
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, BatchEncoding, AutoModel, PreTrainedTokenizer
from typing import Literal, List, Dict, Tuple, Optional
from transformers import BatchEncoding
from autextication import Autextication
from corpus import Corpus
import transformers
from transformers import GPT2Tokenizer
from stats import*


# Corpus mapping
CORPORA: Dict = {

    Autextication.identifier: Autextication
}


class FullDataloader(Dataset):

    def __init__(
            self,
            split: Literal['train', 'validation', 'test']|None,
            cache_dir_path: str,
            tokeniser: PreTrainedTokenizer,
            corpora_path: Optional[str],
            corpus_list: Optional[List[str]],
            corpus_prefix = "NLPdataset",
            task: str = 'mono'):

        self.corpora_path = corpora_path
        self.split = split
        self.corpus_cache_file_path: str = os.path.join(cache_dir_path, f'{corpus_prefix}_{split}_{task}.pbz2')
        self.tokenizer = tokeniser
        self.data: List[Dict]
        self.corpus_list = corpus_list
        self.task = task

        if os.path.exists(self.corpus_cache_file_path):
            self._load_data_from_cache()
        else:
            self._prepare_data()

    def __len__(self) -> int:
        # Number of sequences within the src set
        return len(self.data)

    def __getitem__(self, index: int):
        # Get utterances from src set
        return self.data[index]

    def _prepare_data(self):

        # Create corpora instances

        corpora: List[Corpus] = [
            CORPORA[corpus_id](
                self.split,
                self.corpora_path,
                self.task
            )
            for corpus_id in self.corpus_list
        ]

        datalist = []
        data = []
        for corpus in corpora:
            datalist.append(corpus.load_data())
        for item in datalist:
            data.extend(item)

        self.data = data

        self.labels_to_int()

        # Cache loaded src
        with bz2.BZ2File(self.corpus_cache_file_path, 'wb') as f:
            pickle.dump(self.data, f)

    def _load_data_from_cache(self):
        # Load compressed pickle file
        with bz2.BZ2File(self.corpus_cache_file_path, 'rb') as f:
            self.data = pickle.load(f)

    def collate(
            self, data_for_batches: List[Dict]
    ):

        if self.task == 'mono':
            # Inputs
            input_encodings = self.tokenizer(
                [item['text'] for item in data_for_batches],
                return_tensors='pt',
                padding=True
            )
            # Target outputs
            labels = torch.tensor([item['origin'] for item in data_for_batches])

            return input_encodings, labels

        elif self.task == 'multi':
            # Inputs
            input_encodings = self.tokenizer(
                [item['text'] for item in data_for_batches],
                return_tensors='pt',
                padding=True
            )
            # Target outputs
            origin = torch.tensor([item['origin'] for item in data_for_batches])
            bot = torch.tensor([item['bot'] for item in data_for_batches])
            language = torch.tensor([item['language'] for item in data_for_batches])

            return input_encodings, origin, bot, language

        elif self.task == 'bot':
            # Inputs
            input_encodings = self.tokenizer(
                [item['text'] for item in data_for_batches],
                return_tensors='pt',
                padding=True
            )
            # Target outputs
            bot = torch.tensor([item['bot'] for item in data_for_batches])

            return input_encodings, bot

    def labels_to_int(self):

        for item in self.data:
            if 'origin' in item:
                if item['origin'] == 'human':
                    item['origin'] = 0
                else:
                    item['origin'] = 1
            if 'bot' in item:
                if item['bot'] is not None:
                    if item['bot'] == 'A':
                        item['bot'] = 0
                    elif item['bot'] == 'B':
                        item['bot'] = 1
                    elif item['bot'] == 'C':
                        item['bot'] = 2
                    elif item['bot'] == 'D':
                        item['bot'] = 3
                    elif item['bot'] == 'E':
                        item['bot'] = 4
                    elif item['bot'] == 'F':
                        item['bot'] = 5
                    elif item['bot'] == 'human-generated':
                        item['bot'] = 6
                else:
                    item['bot'] = -1
            if 'language' in item:
                if item['language'] == 'en':
                    item['language'] = 0
                else:
                    item['language'] = 1




tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = AutoModel.from_pretrained('gpt2')


tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_special_tokens({'cls_token': '<|cls|>'})

model.resize_token_embeddings(new_num_tokens=len(tokenizer))
# Extend model
train = FullDataloader('train', '/home/matteo/PycharmProjects/AUTEXTIFICATION/src/cache', tokenizer, '/home/matteo/PycharmProjects/AUTEXTIFICATION/src/rawData', corpus_list=['autextication'], task='bot')
val = FullDataloader('validation', '/home/matteo/PycharmProjects/AUTEXTIFICATION/src/cache', tokenizer, '/home/matteo/PycharmProjects/AUTEXTIFICATION/src/rawData', corpus_list=['autextication'], task='bot')
test = FullDataloader('test', '/home/matteo/PycharmProjects/AUTEXTIFICATION/src/cache', tokenizer, '/home/matteo/PycharmProjects/AUTEXTIFICATION/src/rawData', corpus_list=['autextication'], task='bot')




#test_loader = DataLoader(test, batch_size=1, shuffle=False, collate_fn=test.collate)

test.collate(test.data)
'''
for batch_idx, batch in enumerate(test_loader):
    input_encodings, origin, bot, language = batch
    if language.item() == 5:
        print(f"Batch {batch_idx}:\nInput encodings: {input_encodings}\nOrigin: {origin}\nLanguage: {language}\nBot: {bot}\n")
'''
'''
test.collate(test.data)
import collections



batch_size = 2
test_loader = DataLoader(test, batch_size=batch_size, shuffle=True, collate_fn=test.collate)

# Iterate over the src in the DataLoader object
for batch in test_loader:
    input_encodings,labels = batch

'''




