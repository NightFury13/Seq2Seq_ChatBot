"""
Data loader class that parses the reddit data into a format suitable for the model.
"""
import pickle

class dataLoader(object):
    '''
    Class to load the tsv data files and load vocabulary.
    '''
    def __init__(self, content_filepath, vocab_filepath):
        '''
        class initializer function.
        '''
        self.contexts, self.responses = self.readContentFile(content_filepath) #id-content relations
        self.vocab = self.createVocab(self.contexts, self.responses, vocab_filepath)

    def readContentFile(self, path):
        '''
        read the tsv file contents and create the context and response pairwise lists.
        '''
        contexts = []
        responses = []
        with open(path, 'r') as f:
            for line in f.readlines():
                try:
                    line = [ele.strip('"') for ele in line.strip().split('$')]
                    contexts.append(line[0])
                    responses.append(line[1])
                except:
                    print("[DATALOADER] : Skipped line :",line)
                    continue
        return contexts, responses

    def createVocab(self, contexts, responses, vocab_filepath):
        '''
        create the vocab for specific movie, (general_vocab+movie_vocab)
        '''
        general_vocab = pickle.load(open(vocab_filepath, 'rb'))
        vocab = []
        for line in contexts+responses:
            line = line.split()
            for word in line:
                if word not in vocab:
                    vocab.append(word)
        return vocab
