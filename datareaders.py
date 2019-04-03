from typing import Iterator, List, Dict, Union
import allennlp
from allennlp.data import Vocabulary, DatasetReader, Instance
from allennlp.data.fields import TextField, LabelField
from allennlp.data.tokenizers import Token
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer, ELMoTokenCharactersIndexer
from random import randint
import csv


labels = [  'Background',
            'Uses',
            'Compares or contrasts',
            'Continuation',
            'Future',
            'Motivation']

def label_to_int(label: str):    
    return labels.index(label)

class Csv_elmo_Reader(DatasetReader):
    '''
    Read csv file [id ; label ; citation_context] for elmo embeddings. 
    '''
    
    def __init__(self, token_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__(lazy=False)
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer(), "elmo": ELMoTokenCharactersIndexer() }

    def text_to_instance(self, tokens: List[Token], label: int) -> Instance:
        sentence_field = TextField(tokens, self.token_indexers)
        fields = {"tokens": sentence_field}

        label_field = LabelField(label)
        fields["label"] = label_field

        return Instance(fields)

    def _read(self, file_path: str) -> Iterator[Instance]:
        with open(file_path) as f:
            reader = csv.reader(f)
            first = True
            for row in reader:
                # skip first row
                if first: first= False; continue
                    
                label = str(row[1])

                sentence = row[2]
                yield self.text_to_instance([Token(word) for word in sentence], label)



