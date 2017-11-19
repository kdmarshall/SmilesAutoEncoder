import random
import numpy as np

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
MIN_CHARS = 20
MAX_CHARS = 100 + 2


class DataSet(object):
    def __init__(self, file_path, valid_size=0.2):
        self.file_path = file_path
        self.valid_size = valid_size
        with open(self.file_path) as smi_file:
            self.smiles = np.array(smi_file.read().splitlines())
        self.set_size = self.smiles.shape[0]
        self.total_indices = list(range(self.set_size))
        # random.shuffle(self.total_indices)
        # print(self.smiles_to_dense(self.smiles_list[0]))

    def smiles_to_dense(self, smiles):
        """
            Convert SMILES string to
            dense integer representation.
        """
        char_list = list(smiles)
        encoded = [GO_ID]+[VOCAB.index(c) for c in char_list] + [EOS_ID]
        # encoded = [VOCAB.index(c) for c in char_list]
        encoded_len = len(encoded)
        pad_len = MAX_CHARS - encoded_len
        if pad_len > 0:
            encoded = encoded + [PAD_ID for _ in range(pad_len)]
        return encoded

    def sequence_lengths(self, encoded_batch):
        truth_batch = encoded_batch != PAD_ID
        return np.sum(truth_batch.astype(np.int32), axis=1)

    def get_batch(self, batch_size, batch_type):
        assert batch_type in ('train','valid'), "Unrecognized batch_type %s" % batch_type
        indices = np.random.randint(0, self.set_size, batch_size)
        encoded_smiles_batch = [self.smiles_to_dense(smiles) for smiles in self.smiles[indices]]
        sequence_lens = self.sequence_lengths(np.array(encoded_smiles_batch))
        return encoded_smiles_batch, sequence_lens
    #     def batch_helper(samples, labels, weights):
    #         indices = np.random.randint(0, samples.shape[0], batch_size)
    #         samples_batch = samples[indices]
    #         labels_batch = labels[indices]
    #         weights_batch = weights[indices]
    #         return (samples_batch, labels_batch, weights_batch)

    # if batch_type == 'train':
    #     samples_batch, labels_batch, weights_batch = batch_helper(self.train_samples, self.train_labels, self.train_weights)
    # else:
    #     samples_batch, labels_batch, weights_batch = batch_helper(self.valid_samples, self.valid_labels, self.valid_weights)
    # return (samples_batch, labels_batch, weights_batch)
