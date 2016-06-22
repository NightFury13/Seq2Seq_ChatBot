"""
Data loader class that parses the reddit data into a format suitable for the model.
"""
import pickle
from progressBar import printProgress as PP

class dataLoader(object):
    '''
    Class to load the tsv data files and load vocabulary.
    '''
    def __init__(self, content_filepath, vocab_filepath):
        '''
        class initializer function.
        '''
        self.contexts, self.responses = self.readContentFile(content_filepath) #id-content relations
        self.vocab = self.createVocab(vocab_filepath)

    def readContentFile(self, path):
        '''
        read the tsv file contents and create the context and response pairwise lists.
        '''
        contexts = []
        responses = []
	print("[Data-Loader] : Loading the context-response pairs...")
        with open(path, 'r') as f:
	    lines = f.readlines()
          
            idx = 0
            total = len(lines)
            PP(idx, total, prefix='Progress:', suffix='Complete', barLength=100)
            for line in f.readlines():
            	line = [ele.strip('"') for ele in line.strip().split('$')]
                contexts.append(line[0])
                responses.append(line[1])
		idx += 1
		PP(idx, total, prefix='Progress:', suffix='Complete', barLength=100)
        return contexts, responses

    def createVocab(self, vocab_filepath):
        '''
        create the vocab for specific movie, (general_vocab+movie_vocab)
        '''
        general_vocab = pickle.load(open(vocab_filepath, 'rb'))
        vocab = []
	
	print("[Data-Loader] : Creating vocab from context-response pairs...")
        idx = 0
        total = len(self.contexts+self.responses)
        PP(idx, total, prefix='Progress:', suffix='Complete', barLength=100)
        for line in self.contexts+self.responses:
            line = line.split()
            for word in line:
                if word not in vocab:
                    vocab.append(word)
		idx += 1
		PP(idx, total, prefix='Progress:', suffix='Complete', barLength=100)
        return vocab

