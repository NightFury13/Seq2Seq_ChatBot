# Seq2Seq_ChatBot

### Keras Flavour
I'm assuming the Theano-backend version of Keras. Make sure you have [installed Keras](https://github.com/fchollet/keras#installation).

Apart from this you'll need the following python libraries.

Garbage Collector :   gc (already installed)

Numpy             : `sudo -E pip install numpy`

Pickle            : pickle (aread installed)

### Usage
From the root of the github directory,
```python
$> THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python train.py
```

### IMP Parameters
Inside ~/train.py, there are a set of global-variables,
```python
#------- Global Variables ---------#
content_filepath   = 'new_data/out_TheSimpsons.tsv'
chars = '0123456789+/-*=abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!.,?(){}[]&#_ ' # Char-Level Vocabulary.
TRUNCATE_SIZE = 20000
#----------------------------------#
```
Update these according to your needs (read infrastructure limits). There are other model-definition-parameters in the same 
script but it might not be the best of ideas to change them randomly. Feel free to try out though!
```python
#-------- Parameters for the model and dataset--------#
questions = dataset.contexts[:TRUNCATE_SIZE]
expected = dataset.responses[:TRUNCATE_SIZE]

RNN = recurrent.LSTM
HIDDEN_SIZE = 512
BATCH_SIZE = 10
LAYERS = 3
X_MAXLEN = len(max(questions, key=len))
Y_MAXLEN = len(max(expected, key=len))

ctable = CharacterTable(chars, X_MAXLEN)
#-----------------------------------------------------#
```
