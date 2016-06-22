"""
Vocabulary generator class to create the general vocabulary from data of all movies to give enough context words for training.
"""
from progressBar import printProgress as PP
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
        self.vocab = self.createVocab()

    def createVocab(self):
        '''
        Method to create the overall vocab from a vocab_file.
        '''
        vocab = []
	print("[Gen-Vocab] : Creating the vocabulary...")
        with open(self.vocab_path, 'r') as f:
            lines = f.readlines()

	    idx = 0
	    total = len(lines)
	    PP(idx, total, prefix='Progress:', suffix='Complete', barLength=100)
            for line in lines:
                line = [ele.strip('"') for ele in line.strip().split('$')]
                line = line[0].split()+line[1].split()
                for word in line:
                    if word not in vocab:
                        vocab.append(word)
		idx += 1
	        PP(idx, total, prefix='Progress:', suffix='Complete', barLength=100)
        return vocab

    def saveVocabToFile(self, out_filepath):
        '''
        Save the created dictionary as a pickle file.
        '''
	print "[Gen-Vocab] : Saving vocab to file - ", out_filepath, " ... ",
        pickle.dump(self.vocab, open(out_filepath, 'wb'))
	print "DONE"
