class dataLoader(object):
    def __init__(self, id_filepath, content_filepath):
        self.pc_relations = self.readIDFile(id_filepath) # parent-child relations
        self.ic_relations = self.readContentFile(content_filepath) #id-content relations
        self.vocab = self.createVocab(self.ic_relations)

    def readIDFile(self, path):
        relations = {}
        with open(path,'r') as f:
            for line in f.readlines():
                child, parent = line.strip().split()
                if not parent[0]=='x': #Parent is not a post-id.
                    relations[child] = parent
        return relations

    def readContentFile(self, path):
        relations = {}
        with open(path, 'r') as f:
            for line in f.readlines():
                try:
                    line = line.strip().split()
                    idx = line[0]
                    content = ' '.join(line[1:])
                    relations[idx] = content
                except:
                    print("[DATALOADER] : Skipped line :",line)
                    continue
        return relations

    def createVocab(self, relations):
        vocab = []
        for idx in relations:
            line = relations[idx].split()
            for word in line:
                if word not in vocab:
                    vocab.append(word)
        return vocab
