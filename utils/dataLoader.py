"""
Data loader class that parses the reddit data into a format suitable for the model.
"""
import pickle
from progressBar import printProgress as PP

class dataLoader(object):
    '''
    Class to load the tsv data files and laod context-response pairs.
    '''
    def __init__(self, content_filepath):
        '''
        class initializer function.
        '''
        self.contexts, self.responses = self.readContentFile(content_filepath) #id-content relations

    def readContentFile(self, path):
        '''
        read the tsv file contents and create the context and response pairwise lists.
        '''
        contexts = []
        responses = []
	path_prefix = path.split('/')[-1].split('.')[0]
	print("[Data-Loader] : Loading the context-response pairs...")
	try:
	    with open('parsed_data/'+path_prefix+'_contexts.pkl', 'rb') as f:
		contexts = pickle.load(f)
	    with open('parsed_data/'+path_prefix+'_responses.pkl', 'rb') as f:
		responses = pickle.load(f)
	except:
            with open(path, 'r') as f:
	        lines = f.readlines()
                idx = 0
                total = len(lines)
                PP(idx, total, prefix='Progress:', suffix='Complete', barLength=100)
                for line in lines:
            	    line = [ele.strip('"') for ele in line.strip().split('$')]
                    contexts.append(line[0])
                    responses.append(line[1])
	       	    idx += 1
		    PP(idx, total, prefix='Progress:', suffix='Complete', barLength=100)
		print("[Data-Loader] : Saving context-response as pickles.")
		with open('parsed_data/'+path_prefix+'_contexts.pkl', 'wb') as f:
		    pickle.dump(contexts, f)
		with open('parsed_data/'+path_prefix+'_responses.pkl', 'wb') as f:
		    pickle.dump(responses, f)
        return contexts, responses

