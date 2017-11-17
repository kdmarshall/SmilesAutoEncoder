VOCAB = ('#', '%', '(', ')', '+', '-', '/',
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    '=', '@',
    'A', 'B', 'C', 'F', 'G', 'H', 'I', 'K', 'L', 'M',
    'N', 'O', 'P', 'R', 'S', 'T', 'V', 'W', 'Y', 'Z',
    '[', '\\', ']',
    'a', 'b', 'c', 'd', 'e', 'g', 'h', 'i', 'l', 'n',
    'o', 'p', 'r', 's', 't', 'u'
    '*', 'EOS_ID', 'GO_ID', '<>')

VOCAB_SIZE = len(VOCAB)
UNK_ID = VOCAB_SIZE - 4
EOS_ID = VOCAB_SIZE - 3
GO_ID = VOCAB_SIZE - 2
PAD_ID = VOCAB_SIZE - 1