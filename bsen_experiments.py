import allennlp
from allennlp.common import Params
from allennlp.models import Model
from allennlp.data import Vocabulary, DatasetReader, Instance, DataIterator
from allennlp.training.trainer import Trainer
from typing import Iterator, List, Dict
from allennlp.data.fields import TextField, LabelField
from allennlp.data.tokenizers import Token
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer, ELMoTokenCharactersIndexer
from datareaders import Interec_sent_elmo_Reader, InterecReader
from random import randint
import custom_bcn
import shutil
import csv

NB_CONFIG_FILES = 4
CONFIG_FILE_NAME = 'configs/biatt'     # nb of the file between name and extension
CONFIG_FILE_EXTENSION = '.json'
NB_REPEATS = 4
SER_DIR ='ser_sen'
TRAINSET_PATH = 'data/train_st.csv'
VALSET_PATH = 'data/val_st.csv'

# Training from scratch or using serialized models
restart = input('Delete all serialization directories and stat from scratch ? y/N')
if restart in ['y', 'Y']:
    try:
        shutil.rmtree(SER_DIR)
        print('Serialization directories removed.\n')
    except:
        print('No directory found.\n')
else:
    print('Serialization directories kept.\n')


# reading all data (sentences + abstracts + titles), making vocabulary
dataset_reader = InterecReader()
print('Reading training set...')
trainset = dataset_reader.read(TRAINSET_PATH)
print('Reading validation set...')
devset = dataset_reader.read(VALSET_PATH)
print('Making vocabulary...')
vocab = Vocabulary.from_instances(trainset + devset)

# reading only the sentences in the data, making readers
dataset_reader = Interec_sent_elmo_Reader()
print('Reading training set...')
trainset = dataset_reader.read(TRAINSET_PATH)
print('Reading validation set...')
devset = dataset_reader.read(VALSET_PATH)

for i in range(NB_CONFIG_FILES*NB_REPEATS):
    # Saving vocab
    vocab.save_to_files(SER_DIR+'/'+str(i//NB_REPEATS)+'_'+str(i%NB_REPEATS)+'/vocabulary')
    print('\n---------------------------------------------------------\nConfiguration:',i//NB_REPEATS,'(',i%NB_REPEATS,')','\n')
    
    # Buillding model
    p = Params.from_file(CONFIG_FILE_NAME + str(i//NB_REPEATS) + CONFIG_FILE_EXTENSION)

    iterator = DataIterator.from_params(params=p.pop("iterator"))
    iterator.index_with(vocab)
    print('Loading Model...')
    model = Model.from_params(vocab=vocab, params=p.pop("model"))
    
    trainer = Trainer.from_params(model=model,
                                serialization_dir=(SER_DIR+'/'+str(i//NB_REPEATS)+'_'+str(i%NB_REPEATS)),
                                iterator=iterator,
                                train_data=trainset,
                                validation_data=devset,
                                params=p.pop("trainer"))
    # Training
    print('Training...')
    trainer.train()
