"""
Vocabulary generator class to create the general vocabulary from data of all movies to give enough context words for training.
"""

import pickle

class genVocabCreator(object):
    '''
    Class to create the general vocab.
    '''
    def __init__(self, gen_vocab_filepath):
        '''
        initializer function for the class.
        '''
        self.vocab_path = gen_vocab_filepath
        self.vocab = createVocab()

    def createVocab(self):
        '''
        Method to create the overall vocab from a vocab_file.
        '''
        vocab = []
        with open(self.vocab_path, 'r') as f:
            for line in f.readlines():
                line = [ele.strip('"') for ele in line.strip().split('$')]
                line = line[0].split()+line[1].split()
                for word in line:
                    if word not in vocab:
                        vocab.append(word)
        return vocab

    def saveVocabToFile(self, out_filepath):
        '''
        Save the created dictionary as a pickle file.
        '''
        pickle.dump(self.vocab, open(out_filepath, 'wb'))
