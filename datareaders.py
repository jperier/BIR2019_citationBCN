from typing import Iterator, List, Dict, Union
import allennlp
from allennlp.data import Vocabulary, DatasetReader, Instance
from allennlp.data.fields import TextField, LabelField
from allennlp.data.tokenizers import Token
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer, ELMoTokenCharactersIndexer
from random import randint
import csv


# TODO: keep or remove ## ? tokens et premise

labels = [  'Background',
            'Uses',
            'Compares or contrasts',
            'Continuation',
            'Future',
            'Motivation']

def label_to_int(label: str):    
    return labels.index(label)

class InterecReader(DatasetReader):
    '''
    Read interec csv file for sentence block (no title nor abstract) using elmo embeddings. 
    '''
    
    def __init__(self, token_indexers: Dict[str, TokenIndexer] = None, use_elmo: bool =False) -> None:
        super().__init__(lazy=False)
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer(), "elmo": ELMoTokenCharactersIndexer() }

    def text_to_instance(self, sentences: Union[bool, List[Token]], title_abs: Union[bool, List[Token]], label: int) -> Instance:
        fields = {}
        if sentences:  
            fields["sentences"]= TextField(sentences, self.token_indexers)
        if title_abs:
            fields["t_a"] = TextField(title_abs, self.token_indexers)

        label_field = LabelField(label)
        fields["label"] = label_field

        return Instance(fields)

    # TODO: independent title and abs + source as well
    def _read(self, file_path: str, sentences: bool = True, t_a: bool = True) -> Iterator[Instance]:
        
        with open(file_path) as f:
            reader = csv.reader(f)
            first = True
            for row in reader:
                # skip first row
                if first: first= False; continue
                    
                label = str(row[1])

                if sentences:
                    sentences = row[4].replace('##','').split()
                    sentences = [Token(word) for word in sentences]
                if t_a:
                    t_a = (row[8] + row[9]).split()[:250]
                    t_a = [Token(word) for word in t_a]

                yield self.text_to_instance(sentences, 
                                            t_a, 
                                            label)





class Interec_sent_elmo_Reader(DatasetReader):
    '''
    Read interec csv file for sentence block (no title nor abstract) using elmo embeddings. 
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

                sentence = row[4].replace('##','').split()
                yield self.text_to_instance([Token(word) for word in sentence], label)



class Interec_cross_Reader(DatasetReader):
    '''
    Read interec csv file for cross block with classic embedings (not elmo embeddings). 
    '''
    
    def __init__(self, token_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__(lazy=False)
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}#, "elmo": ELMoTokenCharactersIndexer() }

    def text_to_instance(self, sentence: List[Token], target: List[Token], label: int) -> Instance:
        sentence_field = TextField(sentence, self.token_indexers)
        target_field = TextField(target, self.token_indexers)
        fields = {"premise": sentence_field, "hypothesis": target_field}

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

                sentences = row[4].replace('##','').split()
                title_abs = (row[8] + row[9]).split()[:250]
                yield self.text_to_instance([Token(word) for word in sentences], [Token(word) for word in title_abs], label)





