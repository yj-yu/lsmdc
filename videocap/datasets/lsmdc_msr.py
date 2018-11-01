import numpy as np

from videocap.util import log

import itertools
import os.path
import random
import h5py
import sys

import pandas as pd
import data_util
import hickle as hkl

# For debug purpose

__path__ = os.path.abspath(os.path.dirname(__file__))
eos_word = '<EOS>'


def assert_exists(path):
    assert os.path.exists(path), 'Does not exist : {}'.format(path)

# FIXME
LSMDC_DATA_DIR = os.path.normpath(os.path.join(__path__, '../../dataset/LSMDC'))
assert_exists(LSMDC_DATA_DIR)

DATAFRAME_DIR = os.path.join(LSMDC_DATA_DIR, 'DataFrame')
assert_exists(DATAFRAME_DIR)

VOCABULARY_DIR = os.path.join(LSMDC_DATA_DIR, 'Vocabulary')
assert_exists(VOCABULARY_DIR)

VIDEO_FEATURE_DIR = os.path.join(LSMDC_DATA_DIR, 'LSMDC16_features')
assert_exists(VIDEO_FEATURE_DIR)


class DatasetLSMDC():
    '''
    Access API for LSMDC videos.
    '''

    def __init__(self,
                 dataset_name='train',
                 image_feature_net='resnet',
                 layer='res5c',
                 max_length=80,
                 max_vid_length=40,
                 max_n_videos=None,
                 attr_length=20,
                 wav_data = True,
                 more_data=True,
                 data_type=None):
        self.dataset_name = dataset_name
        self.image_feature_net = image_feature_net
        self.layer = layer
        self.max_length = max_length
        self.max_vid_length = max_vid_length
        self.max_n_videos = max_n_videos
        self.attr_length = attr_length
        self.data_type = data_type
        self.wav_data = wav_data
        self.more_data = more_data
        self.data_df = self.read_df_from_csvfile()

        if max_n_videos is not None:
            self.data_df = self.data_df[:max_n_videos]
        self.ids = list(self.data_df.index)
        self.msr_h5 = h5py.File(os.path.join(VIDEO_FEATURE_DIR,"MSR_RESNET_pool5.hdf5"))

    def __del__(self):
        #self.feat_h5.close()
        self.msr_h5.close()

    def __len__(self):
        if self.max_n_videos is not None:
            if self.max_n_videos <= len(self.data_df):
                return self.max_n_videos
        return len(self.data_df)

    def read_df_from_csvfile(self):
        if self.data_type is None:
            self.data_type = 'CAP'

        assert self.data_type in ['CAP', 'FIB', 'MC'], 'Should choose data type in [CAP, FIB, MC]'

        train_data_path = os.path.join(DATAFRAME_DIR, 'MSR_'+self.data_type+'_train.csv')
        test_data_path = os.path.join(DATAFRAME_DIR, 'MSR_'+self.data_type+'_test.csv')

        train_cap_path = os.path.join(DATAFRAME_DIR, 'MSR_CAP_train.csv')
        test_cap_path = os.path.join(DATAFRAME_DIR, 'MSR_CAP_test.csv')

        if self.dataset_name == 'train':
            if self.data_type in ['MC']:
                train_data_path = train_cap_path
            data_df = pd.read_csv(train_data_path, sep='\t')
        elif self.dataset_name == 'test':
            data_df = pd.read_csv(test_data_path, sep='\t')

        data_df = data_df.set_index('key')
        print ("Data Number : ",len(data_df))
        data_df['row_index'] = range(1, len(data_df)+1)

        return data_df

    @property
    def n_words(self):
        ''' The dictionary size. '''
        if not hasattr(self, 'word2idx'):
            raise Exception('Dictionary not built yet!')
        return len(self.word2idx)

    def __repr__(self):
        if hasattr(self, 'word2idx'):
            return '<DatasetLSMDC (%s) with %d videos and %d words>' % (
                self.dataset_name, len(self), len(self.word2idx))
        else:
            return '<DatasetLSMDC (%s) with %d videos -- dictionary not built>' % (
                self.dataset_name, len(self))

    def split_sentence_into_words(self, sentence):
        '''
        Split the given sentence (str) and enumerate the words as strs.
        Each word is normalized, i.e. lower-cased, non-alphabet characters
        like period (.) or comma (,) are stripped.
        When tokenizing, I use ``data_util.clean_str``
        '''
        try:
            words = data_util.clean_str(sentence).split()
        except:
            print sentence
            sys.exit()
        for w in words:
            if not w:
                continue
            yield w

    def build_word_vocabulary(self):
        word_matrix_path = os.path.join(VOCABULARY_DIR, 'common_word_matrix.hkl')
        assert_exists(word_matrix_path)
        word2idx_path = os.path.join(VOCABULARY_DIR, 'common_word_to_index.hkl')
        assert_exists(word2idx_path)
        idx2word_path = os.path.join(VOCABULARY_DIR, 'common_index_to_word.hkl')
        assert_exists(idx2word_path)

        with open(word_matrix_path, 'r') as f:
            self.word_matrix = hkl.load(f)
        log.info("Load word_matrix from hkl file : %s", word_matrix_path)

        with open(word2idx_path, 'r') as f:
            self.word2idx = hkl.load(f)
        log.info("Load word2idx from hkl file : %s", word2idx_path)

        with open(idx2word_path, 'r') as f:
            self.idx2word = hkl.load(f)
        log.info("Load idx2word from hkl file : %s", idx2word_path)

    def share_word_vocabulary_from(self, dataset):
        assert hasattr(dataset, 'idx2word') and hasattr(dataset, 'word2idx'), \
            'The dataset instance should have idx2word and word2idx'
        assert (isinstance(dataset.idx2word, dict) or isinstance(dataset.idx2word, list)) \
                and isinstance(dataset.word2idx, dict), \
            'The dataset instance should have idx2word and word2idx (as dict)'

        if hasattr(self, 'word2idx'):
            log.warn("Overriding %s' word vocabulary from %s ...", self, dataset)

        self.idx2word = dataset.idx2word
        self.word2idx = dataset.word2idx
        if hasattr(dataset, 'word_matrix'):
            self.word_matrix = dataset.word_matrix

    def iter_ids(self, shuffle=False):
        '''
        Iterate data keys. (e.g. vid123..., FIB456..., MC789...,)
        '''
        if shuffle:
            random.shuffle(self.ids)
        for key in self.ids:
            yield key

    def load_video_feature(self, key):
        video_id = str(self.data_df.loc[key, 'vid_key'])

        if video_id[:3] == 'vid':
            video_feature = np.array(self.feat_h5[video_id])
        elif video_id[:3] == 'msr':
            video_feature = np.array(self.msr_h5[video_id])
        else:
            print "VIDEO_ID",video_id
            raise Exception('video_key error in load_video_feature')

        if self.image_feature_net == 'resnet':
            assert self.layer.lower() in ['pool5']
            if self.layer.lower() == 'pool5':
                video_feature = np.expand_dims(video_feature, axis=1)
                video_feature = np.expand_dims(video_feature, axis=1)
        return video_feature

    def get_video_feature_dimension(self):
        if self.image_feature_net == 'resnet':
            assert self.layer.lower() in ['pool5']
            if self.layer.lower() == 'pool5':
                return (self.max_vid_length, 1, 1, 2048)
        raise NotImplementedError()

    def get_video_feature(self, key):
        video_feature = self.load_video_feature(key)
        return video_feature

    def convert_sentence_to_matrix(self, sentence):
        '''
        Convert the given sentence into word indices and masks.
        WARNING: Unknown words (not in vocabulary) are revmoed.

        Args:
            sentence: A str for unnormalized sentence, containing T words

        Returns:
            sentence_word_indices : list of (at most) length T,
                each being a word index
        '''
        sent2indices = [self.word2idx[w] for w in
                        self.split_sentence_into_words(sentence)
                        if w in self.word2idx]
        T = len(sent2indices)
        length = min(T, self.max_length)
        return sent2indices[:length]

    def get_video_mask(self, video_feature):
        video_length = video_feature.shape[0]
        return data_util.fill_mask(self.max_vid_length,
                                   video_length,
                                   zero_location='LEFT')

    def get_description(self, key):
        '''
        Return caption string for given key.
        '''
        description = self.data_df.loc[key, 'description']
        return self.convert_sentence_to_matrix(description)

    def get_sentence(self, key):
        sentence = self.data_df.loc[key, 'sentence']
        return self.convert_sentence_to_matrix(sentence)

    def get_blank_sentence(self, key):
        blank_sentence = self.data_df.loc[key, 'blank_sentence']
        clean_blank_sent = data_util.clean_blank(blank_sentence)
        sent2indices = [self.word2idx[w] for w in
                        clean_blank_sent
                        if w in self.word2idx]
        T = len(sent2indices)
        length = min(T, self.max_length)
        return sent2indices[:length]

    def get_blank_answer(self, key):
        answer = self.data_df.loc[key, 'answer']
        answer_idx = self.word2idx[data_util.clean_str(answer).split()[0]]
        voc_size = self.word_matrix.shape[0]
        onehot_answer = np.zeros(voc_size)
        onehot_answer[answer_idx] = 1
        return onehot_answer

    def get_sentence_mask(self, sentence):
        sent_length = len(sentence)
        return data_util.fill_mask(self.max_length,
                                   sent_length,
                                   zero_location='RIGHT')

    def get_blank_sent_mask(self, sentence):
        mask = np.zeros(self.max_length)
        if 1 in sentence:
            idx = sentence.index(1)
            mask[idx] = 1
        return mask

    def get_reverse_blank_sent_mask(self,sentence):
        mask = np.zeros(self.max_length)
        for i,idx in enumerate(sentence):
            if idx!=1:
                mask[i] = 1
        return mask

    def get_MC_dict(self, key):
        if self.dataset_name == 'train':
            a1 = self.data_df.loc[key, 'description']
            keys = np.random.choice(self.ids, 4, replace=False)
            a2 = self.data_df.loc[keys[0], 'description']
            a3 = self.data_df.loc[keys[1], 'description']
            a4 = self.data_df.loc[keys[2], 'description']
            a5 = self.data_df.loc[keys[3], 'description']
            answer = 0
        else:
            a1 = self.data_df.loc[key, 'a1']
            a2 = self.data_df.loc[key, 'a2']
            a3 = self.data_df.loc[key, 'a3']
            a4 = self.data_df.loc[key, 'a4']
            a5 = self.data_df.loc[key, 'a5']
            answer = self.data_df.loc[key, 'answer']
        row_index = self.data_df.loc[key, 'row_index']

        # as list of sentence strings
        candidates = [a1, a2, a3, a4, a5]
        candidates_to_indices = [self.convert_sentence_to_matrix(x)
                                 for x in candidates]
        return {
            'answer': answer,
            'candidates': candidates_to_indices,
            'raw_sentences': candidates,
            'row_indices': row_index,
        }

    def get_MC_matrix(self, candidates):
        candidates_matrix = np.zeros([5, self.max_length], dtype=np.uint32)
        for k in range(5):
            sentence = candidates[k]
            candidates_matrix[k, :len(sentence)] = sentence
        return candidates_matrix

    def get_MC_mask(self, candidates):
        mask_matrix = np.zeros([5, self.max_length], dtype=np.uint32)
        for k in range(5):
            mask_matrix[k] = data_util.fill_mask(self.max_length,
                                                 len(candidates[k]),
                                                 zero_location='RIGHT')
        return mask_matrix

    def assemble_into_sentence(self, word_matrix):
        '''
        Convert the word matrix (Batch x MaxLength) into a list of
        human-readable setnences, w.r.t the current directory.
        '''
        B, T = word_matrix.shape
        sentences = [None] * B

        for b in xrange(B):
            if 2 in word_matrix[b]:
                eos_position = list(word_matrix[b]).index(2)
            else:
                eos_position = len(word_matrix[b])

            sent = ' '.join(self.idx2word[int(i)] for i in word_matrix[b, :eos_position])
            sentences[b] = sent

        return sentences

    def get_CAP_result(self, chunk):
        batch_size = len(chunk)
        batch_video_feature_convmap = np.zeros([batch_size]
                                               + list(self.get_video_feature_dimension()),
                                               dtype=np.float32)
        batch_caption = np.zeros([batch_size, self.max_length], dtype=np.uint32)

        batch_video_mask = np.zeros([batch_size, self.max_vid_length], dtype=np.uint32)
        batch_caption_mask = np.zeros([batch_size, self.max_length], dtype=np.uint32)

        batch_debug_sent = np.asarray([None] * batch_size)

        for k in xrange(batch_size):
            key = chunk[k]
            video_feature = self.get_video_feature(key)
            video_mask = self.get_video_mask(video_feature)

            batch_video_feature_convmap[k] = data_util.pad_video(video_feature,
                                                                    self.get_video_feature_dimension())
            batch_video_mask[k] = video_mask

            if self.dataset_name != 'blind':
                try:
                    caption = self.get_description(key)
                    caption_mask = self.get_sentence_mask(caption)
                except:
                    print key
                    sys.exit()
                batch_caption[k, :len(caption)] = caption
                batch_caption_mask[k] = caption_mask
                batch_debug_sent[k] = self.data_df.loc[key, 'description']

        ret = {
            'ids': chunk,
            'video_features': batch_video_feature_convmap,
            'caption_words': batch_caption,
            'video_mask': batch_video_mask,
            'caption_mask': batch_caption_mask,
            'debug_sent': batch_debug_sent
        }
        return ret

    def get_FIB_result(self, chunk):
        batch_size = len(chunk)
        batch_video_feature_convmap = np.zeros([batch_size]
                                               + list(self.get_video_feature_dimension()),
                                               dtype=np.float32)
        batch_blank_sent = np.zeros([batch_size, self.max_length], dtype=np.uint32)
        batch_answer = np.zeros([batch_size, self.word_matrix.shape[0]], dtype=np.uint32)

        batch_video_mask = np.zeros([batch_size, self.max_vid_length], dtype=np.uint32)
        batch_blank_sent_mask = np.zeros([batch_size, self.max_length], dtype=np.uint32)
        batch_reverse_blank_sent_mask = np.zeros([batch_size, self.max_length], dtype=np.uint32)
        batch_debug_sent = np.asarray([None] * batch_size)

        for k in xrange(batch_size):
            key = chunk[k]
            video_feature = self.get_video_feature(key)
            blank_sent = self.get_blank_sentence(key)
            answer = self.get_blank_answer(key)

            video_mask = self.get_video_mask(video_feature)
            blank_sent_mask = self.get_blank_sent_mask(blank_sent)
            reverse_blank_sent_mask = self.get_reverse_blank_sent_mask(blank_sent)

            batch_video_feature_convmap[k] = data_util.pad_video(video_feature,
                                                                    self.get_video_feature_dimension())

            batch_blank_sent[k, :len(blank_sent)] = blank_sent
            batch_answer[k] = answer

            batch_video_mask[k] = video_mask
            batch_blank_sent_mask[k] = blank_sent_mask
            batch_reverse_blank_sent_mask[k] = reverse_blank_sent_mask

            batch_debug_sent[k] = self.data_df.loc[key, 'sentence']

        ret = {
            'ids': chunk,
            'video_features': batch_video_feature_convmap,
            'blank_sent': batch_blank_sent,
            'answer': batch_answer,
            'video_mask': batch_video_mask,
            'answer': batch_answer,
            'blank_sent_mask': batch_blank_sent_mask,
            'debug_sent': batch_debug_sent,
            'reverse_blank_sent_mask': batch_reverse_blank_sent_mask
        }
        return ret

    def get_MC_result(self, chunk):
        batch_size = len(chunk)
        batch_video_feature_convmap = np.zeros([batch_size]
                                               + list(self.get_video_feature_dimension()),
                                               dtype=np.float32)
        batch_candidates = np.zeros([batch_size, 5, self.max_length], dtype=np.uint32)
        batch_answer = np.zeros([batch_size], dtype=np.uint32)

        batch_video_mask = np.zeros([batch_size, self.max_vid_length], dtype=np.uint32)
        batch_candidates_mask = np.zeros([batch_size, 5, self.max_length], dtype=np.uint32)

        batch_debug_sent = np.asarray([None] * batch_size)
        batch_raw_sentences = np.asarray([[None]*5 for _ in range(batch_size)])
        batch_row_indices = np.asarray([-1] * batch_size)

        for k in xrange(batch_size):
            key = chunk[k]

            MC_dict = self.get_MC_dict(key)
            candidates = MC_dict['candidates']
            raw_sentences = MC_dict['raw_sentences']
            answer = MC_dict['answer']

            video_feature = self.get_video_feature(key)
            candidates_matrix = self.get_MC_matrix(candidates)

            video_mask = self.get_video_mask(video_feature)
            candidates_mask = self.get_MC_mask(candidates)

            batch_video_feature_convmap[k] = data_util.pad_video(video_feature,
                                                                    self.get_video_feature_dimension())
            batch_candidates[k] = candidates_matrix
            batch_raw_sentences[k, :] = raw_sentences
            batch_answer[k] = 0
            batch_video_mask[k] = video_mask
            batch_candidates_mask[k] = candidates_mask
            batch_row_indices[k] = MC_dict['row_indices']

            if answer != 0:
                batch_candidates[k, [0, answer], :] = batch_candidates[k, [answer, 0], :]
                batch_candidates_mask[k, [0, answer], :] = batch_candidates_mask[k, [answer, 0], :]
                batch_raw_sentences[k, [0, answer]] = batch_raw_sentences[k, [answer, 0]]


        ret = {
            'ids': chunk,
            'video_features': batch_video_feature_convmap,
            'candidates': batch_candidates,
            'raw_sentences': batch_raw_sentences,
            'answer': batch_answer,
            'video_mask': batch_video_mask,
            'candidates_mask': batch_candidates_mask,
            'row_indices': batch_row_indices
        }
        return ret

    def next_batch(self, batch_size=64, include_extra=False, shuffle=True):
        if not hasattr(self, '_batch_it'):
            self._batch_it = itertools.cycle(self.iter_ids(shuffle=shuffle))

        chunk = []
        for k in xrange(batch_size):
            key = next(self._batch_it)
            chunk.append(key)

        if self.data_type == 'CAP':
            return self.get_CAP_result(chunk)
        elif self.data_type == 'FIB':
            return self.get_FIB_result(chunk)
        elif self.data_type == 'MC':
            return self.get_MC_result(chunk)
        else:
            raise Exception('data_type error in next_batch')

    def batch_iter(self, num_epochs, batch_size, shuffle=True):
        for epoch in xrange(num_epochs):
            steps_in_epoch = int(len(self) / batch_size)
            self._batch_it = itertools.cycle(self.iter_ids(shuffle=shuffle))
            for s in range(steps_in_epoch+1):
                yield self.next_batch(batch_size,
                                      shuffle=shuffle)

    def batch_tile(self, num_epochs, batch_size, neg=True):
        keys = self.ids
        for epoch in xrange(num_epochs):
            for i in range(int(1000/batch_size)):
                for j in range(int(1000/batch_size)):
                    y_keys = keys[i*batch_size: (i+1)*batch_size]
                    x_keys = keys[j*batch_size: (j+1)*batch_size]
                    yield self.next_tile(batch_size, y_keys, x_keys)


