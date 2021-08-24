#%% ##############################second part: load data, and train the model############################ 
import pickle
from flair.data import Corpus
from flair.datasets import SentenceDataset
from sklearn.model_selection import train_test_split
from flair.embeddings import WordEmbeddings, CharacterEmbeddings,StackedEmbeddings
#%%
from KD_sequence_tagger_model import SequenceTagger
# from flair.models import SequenceTagger

path = '/Users/Wu/Google Drive/'
with open(path+'data/data_LogitsLabel_25k.pickle', 'rb') as handle:
    data = pickle.load(handle)

data_train, data_test = train_test_split(data, test_size=0.2, random_state=42)
data_dev,data_test = train_test_split(data_test, test_size=0.5, random_state=42)
corpus: Corpus = Corpus(SentenceDataset(data_train),SentenceDataset(data_test),SentenceDataset(data_dev))
tag_type = 'ner'
tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)
corpus: Corpus = corpus.downsample(0.1)

embedding_types = [
    WordEmbeddings('glove'),
    CharacterEmbeddings(),
]
embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)

tagger: SequenceTagger = SequenceTagger(hidden_size=256,
                                        embeddings=embeddings,
                                        tag_dictionary=tag_dictionary,
                                        tag_type=tag_type,
                                        use_crf=True,
                                        reproject_embeddings=50,
                                        use_soft_labels=True,
                                        # use_logits = True,
                                        )

#%% train the model 
from flair.trainers import ModelTrainer
trainer: ModelTrainer = ModelTrainer(tagger, corpus)

trainer.train('resources/taggers/test2.5k_softlabels_crf',
              learning_rate=0.1,
              mini_batch_size=10,
              max_epochs=15,
              checkpoint=True,
              )
#%% continue training the model
# checkpoint = 'resources/taggers/test/checkpoint.pt'
# trainer = ModelTrainer.load_checkpoint(checkpoint, corpus) 
# trainer.train(path+'resources/taggers/test60ep',
#               learning_rate=0.05,
#               mini_batch_size=10,
#               max_epochs=50,
#               checkpoint=True) 

