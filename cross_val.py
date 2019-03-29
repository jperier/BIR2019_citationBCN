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
from random import randint, shuffle
import custom_bcn
import shutil
import csv
import os

# k-fold cross-validation
K = 10

NB_CONFIG_FILES = 3
CONFIG_FILE_NAME = 'configs/biatt'     # nb of the file between name and extension
CONFIG_FILE_EXTENSION = '.json'
NB_REPEATS = 3
SER_DIR ='ser'
DATASET_PATH = 'data/stan_teufel.csv'
# The names you want the files to have
TRAINSET_PATH = 'data/train_st_crossval.csv'
VALSET_PATH = 'data/val_st_crossval.csv'
CLEAN_SER = True # to removed serialized models (except best) in order to save disk space

def split_train_test(iteration, K):
    d = {}

    first = True
    with open(DATASET_PATH, 'r') as f:

        reader = csv.reader(f)
        length = 0
        for row in reader:
            if first: first = False; continue
            if row[2] not in d.keys():
                d[row[2]] = [row]
            else:
                d[row[2]].append(row)
            length+=1

    articles = []
    for i in range(len(d)):
        max = 0
        kmax = None
        for k in d.keys():
            if len(d[k]) > max:
                max = len(d[k])
                kmax = k
        articles.append(d.pop(kmax))
    
    sets = []
    for i in range(K):
        sets.append([])
    for i,a in enumerate(articles):
        for row in a:
            sets[i%K].append(row)
    train = []
    val = []
    for i in range(K):
        if i == iteration:
            val = sets[i]
        else:
            train += sets[i]
    
    print('val:',len(val), 'train:', len(train))

    shuffle(train)

    with open(TRAINSET_PATH, 'w') as f:
        writer = csv.writer(f)
        for row in train:
            writer.writerow(row)

    with open(VALSET_PATH, 'w') as f:
        writer = csv.writer(f)
        for row in val:
            writer.writerow(row)





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

# reading data, making vocabulary
dataset_reader = InterecReader()
print('Reading dataset...')
dataset = dataset_reader.read(DATASET_PATH)
vocab = Vocabulary.from_instances(dataset)


for i in range(NB_CONFIG_FILES*NB_REPEATS):

    for j in range(K):
        split_train_test(j, K)
        dataset_reader = Interec_sent_elmo_Reader()
        print('Reading training set...')
        trainset = dataset_reader.read(TRAINSET_PATH)
        print('Reading validation set...')
        devset = dataset_reader.read(VALSET_PATH)

        # Saving vocab in serdir
        vocab.save_to_files(SER_DIR+'/'+str(i//NB_REPEATS)+'_'+str(i%NB_REPEATS)+'/vocabulary')

        # Building model
        print('\n---------------------------------------------------------\nConfiguration:',i//NB_REPEATS,'(',i%NB_REPEATS,')','\n')
        p = Params.from_file(CONFIG_FILE_NAME + str(i//NB_REPEATS) + CONFIG_FILE_EXTENSION)

        iterator = DataIterator.from_params(params=p.pop("iterator"))
        iterator.index_with(vocab)
        print('Loading Model...')
        model = Model.from_params(vocab=vocab, params=p.pop("model"))
        
        trainer = Trainer.from_params(model=model,
                                    serialization_dir=(SER_DIR+'/'+str(i//NB_REPEATS)+'_'+str(i%NB_REPEATS)+'_'+str(j)),
                                    iterator=iterator,
                                    train_data=trainset,
                                    validation_data=devset,
                                    params=p.pop("trainer"))

        # Training
        print('Training...')
        trainer.train()

        # removing serialized models (except best) to save disk space
        if (CLEAN_SER):
            serdir = (SER_DIR+'/'+str(i//NB_REPEATS)+'_'+str(i%NB_REPEATS)+'_'+str(j))
            nb = 0
            while os.path.isfile(serdir + '/training_state_epoch_' + str(nb) + '.th'):
                os.remove(serdir + '/training_state_epoch_' + str(nb) + '.th')
                nb+=1
            nb = 0
            while os.path.isfile(serdir + '/model_state_epoch_' + str(nb) + '.th'):
                os.remove(serdir + '/model_state_epoch_' + str(nb) + '.th')
                nb+=1
	
