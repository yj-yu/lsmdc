from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
import numpy as np
import time
import math
import hickle as hkl

from tensorflow.contrib.rnn import static_bidirectional_rnn
from tensorflow.contrib.layers import batch_norm

from videocap.util import log
from videocap.utils import common_attention
from videocap.datasets import data_util

import tensorflow.contrib.slim as slim
import tensorflow.contrib.rnn as rnn_cell


_lstm_map = {
    'BasicLSTM': rnn_cell.BasicLSTMCell,
}


class FIBGenerator(object):
    """
    """
    def __init__(self, config, word_embed):
        self.config = config
        self.batch_size = config.batch_size
        self.word_embed = word_embed
        self.vocab_size = word_embed.shape[0]
        self.name = 'Fill-In-the-BlankGenerator'

        self.dropout_keep_prob = tf.placeholder_with_default(
            self.config.dropout_keep_prob, [])

        self.batch_size = config.batch_size
        self.video_steps = config.video_steps
        self.initializer = tf.contrib.layers.xavier_initializer(
            uniform=False)
        self.video_cell = lambda:  _lstm_map[self.config.lstm_cell](
            num_units=self.config.hidden_dim, state_is_tuple=True)
        self.caption_cell = lambda: _lstm_map[self.config.lstm_cell](
            num_units=self.config.hidden_dim, state_is_tuple=True)

    def get_feed_dict(self, batch_chunk):
        feed_dict = {
            self.video: batch_chunk['video_features'].astype(float),
            self.video_mask: batch_chunk['video_mask'].astype(float),
            self.blank_caption: batch_chunk['blank_sent'].astype(float),
            self.blank_caption_mask: batch_chunk['blank_sent_mask'].astype(float),
            self.answer: batch_chunk['answer'],
            self.reverse_blank_sent_mask : batch_chunk['reverse_blank_sent_mask'].astype(float)
        }
        return feed_dict

    def get_placeholder(self):
        if self.config.image_feature_net == 'resnet':
            if self.config.wav_data:
                video = tf.placeholder(tf.float32, [self.config.batch_size, self.config.video_steps, 1, 1, 2176])
            else:
                video = tf.placeholder(tf.float32, [self.config.batch_size, self.config.video_steps, 1, 1, 2048])
        elif self.config.image_feature_net == 'vgg':
            video = tf.placeholder(tf.float32, [self.config.batch_size, self.config.video_steps, 1, 1, 4096])

        video_mask = tf.placeholder(tf.float32, [self.config.batch_size, self.config.video_steps])
        blank_caption = tf.placeholder(tf.int32, [self.config.batch_size, self.config.caption_length])
        blank_caption_mask = tf.placeholder(tf.float32, [self.config.batch_size, self.config.caption_length])
        answer = tf.placeholder(tf.float32, [self.config.batch_size, self.vocab_size])
        train_flag = tf.placeholder(tf.bool)
        reverse_blank_sent_mask = tf.placeholder(tf.float32, [self.config.batch_size, self.config.caption_length])

        result = {
            'video': video,
            'video_mask': video_mask,
            'blank_caption': blank_caption,
            'blank_caption_mask': blank_caption_mask,
            'answer': answer,
            'train_flag': train_flag,
            'reverse_blank_sent_mask': reverse_blank_sent_mask
        }
        return result

    def build_caption_embedding(self, input_seqs, name=None, reuse_variable=False):
        """Builds the input sequence(caption) embeddings.

        Inputs:
            self.input_seqs

        Outputs:
            seq_embeddings
        """
        with tf.variable_scope("seq_embedding", reuse=reuse_variable), tf.device("/cpu:0"):
            seq_embeddings = tf.nn.embedding_lookup(self.word_embed_t, input_seqs, name=name)

        return seq_embeddings


    def build_video_embedding(self, video_cell, video, video_mask, reuse_variable):
        video = tf.reduce_mean(video, [2, 3])
        video = common_attention.add_timing_signal_nd(video)
        video_emb = video * tf.expand_dims(video_mask, 2)

        with tf.variable_scope("video_rnn", reuse=reuse_variable) as scope:
            with slim.arg_scope([slim.conv2d],
                                        weights_initializer=self.initializer,
                                        weights_regularizer=slim.l2_regularizer(0.0005)):
                video_emb = tf.nn.dropout(video_emb,self.dropout_keep_prob)

                #video : [BxLx1x2048]
                video_emb_exp = tf.expand_dims(video_emb, 2)
                conv1 = slim.conv2d(video_emb_exp, 2048, [3, 1],padding='SAME', scope='conv1')
                conv2 = slim.conv2d(conv1, 2048, [3, 1],padding='SAME', scope='conv2')
                conv3 = slim.conv2d(conv2, 2048, [3, 1],padding='SAME', scope='conv3')
                outputs = tf.reduce_mean(conv3, 2)
                'gated_linear_units'
                input_pass = outputs[:,:,0:1024]
                input_gate = outputs[:,:,1024:]
                input_gate = tf.sigmoid(input_gate)
                outputs =  tf.multiply(input_pass, input_gate)

            outputs = tf.nn.dropout(outputs,self.dropout_keep_prob)
            masked_outputs = outputs * tf.expand_dims(video_mask, 2)
        masked_outputs = slim.fully_connected(masked_outputs, 1024, scope='vid_rnn_fc', activation_fn=tf.nn.tanh, normalizer_fn=slim.batch_norm, normalizer_params={'is_training':self.train_flag})


        return masked_outputs


    def leak_relu(self,x,leak=0.2):
        f1 = 0.5*(1+leak)
        f2 = 0.5*(1-leak)
        return f1*x+f2*abs(x)

    def build_caption_encoder(self,
                              caption_cell,
                              reuse_variable=False):
        embedded_sentence = self.build_caption_embedding(self.blank_caption, name="embedding_sent")
        embedded_sentence = embedded_sentence * tf.expand_dims(self.reverse_blank_sent_mask, 2)
        embedded_sentence = tf.nn.dropout(embedded_sentence, self.dropout_keep_prob)

        with tf.variable_scope("attended_input", reuse=reuse_variable) as scope:
            self.sentence_list = []

            for i in range(self.config.caption_length):
                if i > 0:
                    scope.reuse_variables()
                attended_sentence = embedded_sentence[:, i, :]
                self.sentence_list.append(attended_sentence)

        with tf.variable_scope("cap_rnn", reuse=reuse_variable) as scope:
            outputs, _, _ = tf.contrib.rnn.static_bidirectional_rnn(caption_cell[0],
                                                                    caption_cell[1],
                                                                    self.sentence_list,
                                                                    dtype=tf.float32)
            outputs = tf.stack(outputs)
            outputs = tf.transpose(outputs, [1, 0, 2])

            outputs = slim.fully_connected(outputs, 1024, scope='cap_rnn_fc', activation_fn=tf.nn.tanh, normalizer_fn=slim.batch_norm, normalizer_params={'is_training':self.train_flag})
            outputs = tf.nn.dropout(outputs,self.dropout_keep_prob)
            masked_outputs = outputs * tf.expand_dims(self.blank_caption_mask, 2)
            rnn_output = tf.reduce_sum(masked_outputs,1)
        return rnn_output

    def build_fib_decoder(self,
                          vid_emb_output,
                          cap_emb_output,
                          reuse_variable=False):
        with tf.variable_scope("fusion", initializer=self.initializer) as scope:
            states = []
            for i in range(int(self.config.video_steps)):
                sum_repr = tf.multiply(vid_emb_output[:,i,:], cap_emb_output)
                states.append(sum_repr)
            fuse = tf.stack(states, axis=1)
        # B x V x 1 x 1024
        fuse = tf.expand_dims(fuse,2);


        with tf.variable_scope("after_fusion", initializer=self.initializer) as scope:
            with slim.arg_scope([slim.fully_connected],
                                weights_regularizer=slim.l2_regularizer(0.0005),
                                normalizer_fn=slim.batch_norm,
                                normalizer_params={'is_training':self.train_flag},
                                reuse=False):
                h1 = slim.fully_connected(fuse, 1024, activation_fn=tf.nn.tanh, scope='fc1')
                input_gate1 = slim.fully_connected(h1, 1, activation_fn=tf.nn.sigmoid, scope='fcalp')

                h2 = slim.fully_connected(fuse, 1024, activation_fn=tf.nn.tanh, scope='fc2')
                input_pass1 = slim.fully_connected(h2, 1024, activation_fn=tf.nn.tanh, scope='fc3')
                output1 = tf.multiply(input_gate1, input_pass1)
                output1 = tf.multiply(output1, tf.expand_dims(tf.expand_dims(self.video_mask,2),3))

            with slim.arg_scope([slim.conv2d],
                                weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                weights_regularizer=slim.l2_regularizer(0.0005),
                                activation_fn = tf.nn.tanh):
                conv1 = slim.conv2d(output1, 1025, [3, 1],padding='Valid', scope='conv1')
                input_pass2 = conv1[:,:,:,:1024]
                input_gate2 = tf.nn.sigmoid(conv1[:,:,:,1024:])
                output2 = tf.multiply(input_pass2,input_gate2)
                output2 = tf.multiply(output2, tf.expand_dims(tf.expand_dims(self.video_mask_list[0],2),3))

                conv2 = slim.conv2d(output2, 1025, [3, 1],padding='Valid', scope='conv2')
                input_pass3 = conv2[:,:,:,:1024]
                input_gate3 = tf.nn.sigmoid(conv2[:,:,:,1024:])
                output3 = tf.multiply(input_pass3,input_gate3)
                output3 = tf.multiply(output3, tf.expand_dims(tf.expand_dims(self.video_mask_list[1],2),3))

                conv3 = slim.conv2d(output3, 1025, [3, 1],[2,1],padding='Valid', scope='conv3')
                input_pass4 = conv3[:,:,:,:1024]
                input_gate4 = tf.nn.sigmoid(conv3[:,:,:,1024:])
                output4 = tf.multiply(input_pass4,input_gate4)
                output4 = tf.multiply(output4, tf.expand_dims(tf.expand_dims(self.video_mask_list[2],2),3))

            sum_output = tf.reduce_sum(output4,[1,2])

        with tf.variable_scope("after_sum", initializer=self.initializer) as scope:
            with slim.arg_scope([slim.fully_connected],
                                weights_regularizer=slim.l2_regularizer(0.0005),
                                normalizer_fn=slim.batch_norm,
                                normalizer_params={'is_training':self.train_flag},
                                reuse=False):
                logit = slim.fully_connected(sum_output, 1024, activation_fn=tf.nn.tanh,scope='fc1')
                logit = slim.fully_connected(logit, 1024, activation_fn=tf.nn.tanh, scope='fc2')
                logit = slim.fully_connected(logit, 1024, activation_fn=tf.nn.tanh, scope='fc3')

        single_emb = (cap_emb_output+logit)/2.
        with tf.variable_scope("rnn_output") as scope:
            scores = slim.fully_connected(inputs = single_emb,
                                           num_outputs = self.vocab_size,
                                           activation_fn=None,
                                           weights_initializer=self.initializer,
                                           scope=scope)
            predictions = tf.argmax(scores, 1)
        with tf.variable_scope("acc_and_loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=scores, labels=self.answer, name="word_loss")
            losses = tf.reduce_mean(losses)
            self.correct_predictions = tf.equal(predictions, tf.argmax(self.answer, 1))
            acc = tf.reduce_mean(tf.cast(self.correct_predictions, "float"), name="accuracy")

        self.input_gate = [input_gate1, input_gate2, input_gate3, input_gate4]
        return losses, predictions, acc, scores

    def build_model(self,
                    video,
                    video_mask,
                    blank_caption,
                    blank_caption_mask,
                    answer,
                    train_flag,
                    reverse_blank_sent_mask,
                    reuse_variable=False):

        self.video = video
        self.video_mask = video_mask
        self.blank_caption = blank_caption
        self.blank_caption_mask = blank_caption_mask
        self.answer = answer
        self.train_flag = train_flag
        self.reverse_blank_sent_mask = reverse_blank_sent_mask

        #Mask list
        video_mask_leng = tf.cast(tf.reduce_sum(self.video_mask,1),tf.int32)
        self.video_mask_list = []
        max_len = self.config.video_steps
        for mi in range(2):
            video_mask_leng = tf.maximum(1, video_mask_leng-2)
            max_len -= 2
            self.video_mask_list.append(tf.reverse(tf.sequence_mask(video_mask_leng,max_len,tf.float32),[-1]))
        max_len = int((max_len-1)/2)
        video_mask_leng = tf.cast((video_mask_leng-1)/2,tf.int32)
        video_mask_leng = tf.maximum(1,video_mask_leng)
        self.video_mask_list.append(tf.reverse(tf.sequence_mask(video_mask_leng,max_len,tf.float32),[-1]))


        self.word_embed_t = tf.Variable(self.word_embed, dtype=tf.float32, name="word_embed", trainable=True)


        self.video_cell_d = lambda: rnn_cell.DropoutWrapper(
                self.video_cell(),
                input_keep_prob = self.dropout_keep_prob,
                output_keep_prob = self.dropout_keep_prob)
        self.caption_cell_d = lambda: rnn_cell.DropoutWrapper(
                self.caption_cell(),
                input_keep_prob = self.dropout_keep_prob,
                output_keep_prob = self.dropout_keep_prob)

        self.video_cell1 = rnn_cell.MultiRNNCell([self.video_cell_d() for _ in range(self.config.num_layers)],
                                                state_is_tuple=True)
        self.video_cell2 = rnn_cell.MultiRNNCell([self.video_cell_d() for _ in range(self.config.num_layers)],
                                                state_is_tuple=True)
        self.video_cell3 = rnn_cell.MultiRNNCell([self.video_cell_d() for _ in range(self.config.num_layers)],
                                                state_is_tuple=True)
        self.video_cell = [self.video_cell1, self.video_cell2, self.video_cell3]

        self.caption_cell1 = rnn_cell.MultiRNNCell([self.caption_cell_d() for _ in range(self.config.num_layers)],
                                                state_is_tuple=True)
        self.caption_cell2 = rnn_cell.MultiRNNCell([self.caption_cell_d() for _ in range(self.config.num_layers)],
                                                state_is_tuple=True)
        self.caption_cell = [self.caption_cell1, self.caption_cell2]


        vid_emb_output= self.build_video_embedding(self.video_cell,self.video, self.video_mask, reuse_variable)

        rnn_emb_output = self.build_caption_encoder(self.caption_cell, reuse_variable)

        self.fib_loss, self.predictions, self.acc, self.logits  = self.build_fib_decoder(vid_emb_output,rnn_emb_output)
        self.concept_loss = tf.constant(0)

        self.mean_loss = self.fib_loss


class FIBTrainer(object):

    def __init__(self, config, model, sess=None, train_summary_dir=None):
        self.sess = sess or tf.get_default_session()
        self.model = model
        self.config = config
        self.train_summary_dir = train_summary_dir
        self.global_step = tf.Variable(0, name="global_step", trainable=False)

        learning_rate_decay_fn = None
        if self.config.learning_rate_decay_factor > 0 and self.config.learning_rate_decay_steps > 0:
            def _learning_rate_decay_fn(learning_rate, global_step):
                return tf.train.exponential_decay(
                    learning_rate, global_step,
                    decay_steps=self.config.learning_rate_decay_steps,
                    decay_rate=self.config.learning_rate_decay_factor,
                    staircase=True)
            learning_rate_decay_fn = _learning_rate_decay_fn

        self.no_op = tf.no_op()
        self.train_op = tf.contrib.layers.optimize_loss(
            loss=self.model.mean_loss,
            global_step=self.global_step,
            learning_rate=self.config.learning_rate,
            learning_rate_decay_fn=learning_rate_decay_fn,
            optimizer=self.config.optimizer,
            clip_gradients=self.config.max_grad_norm,
            summaries=["learning_rate"]
        )
        self.summary_mean_loss = tf.summary.scalar("mean_loss", model.mean_loss)
        self.train_summary_writer = None
        if train_summary_dir is not None:
            self.train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph_def)

        self.build_eval_graph()

    def build_eval_graph(self):
        self.total_correct = tf.Variable(0.0, trainable=False, collections=[], name="total_correct")
        self.example_count = tf.Variable(0.0, trainable=False, collections=[], name="example_count")

        self.accuracy = self.total_correct / self.example_count
        inc_total_correct = self.total_correct.assign_add(
            tf.reduce_sum(tf.cast(self.model.correct_predictions, "float")))
        inc_example_count = self.example_count.assign_add(self.model.batch_size)

        with tf.control_dependencies([self.total_correct.initializer,
                                      self.example_count.initializer]):
            self.eval_reset = tf.no_op()

        with tf.control_dependencies([inc_total_correct, inc_example_count]):
            self.eval_step = tf.no_op()

    def run_single_step(self, queue, is_train=True):
        start_ts = time.time()

        step_op = self.train_op if is_train else self.no_op
        batch_chunk = queue.get_inputs()
        feed_dict = self.model.get_feed_dict(batch_chunk)
        feed_dict[self.model.train_flag] = is_train
        if not is_train: feed_dict[self.model.dropout_keep_prob] = 1.0

        _, loss, acc, concept_loss, current_step, summary = self.sess.run(
            [step_op, self.model.mean_loss, self.model.acc, self.model.concept_loss, self.global_step,
             self.summary_mean_loss],
            feed_dict=feed_dict)
        end_ts = time.time()
        result = {
            "loss": loss,
            "concept_loss": concept_loss,
            "acc": acc,
            "current_step": current_step,
            "step_time": (end_ts - start_ts)
        }
        return result

    def eval_single_step(self, val_queue):
        batch_chunk = val_queue.get_inputs()
        feed_dict = self.model.get_feed_dict(batch_chunk)
        feed_dict[self.model.train_flag] = False
        feed_dict[self.model.dropout_keep_prob] = 1.0
        _, loss, predictions, logits, input_gate = self.sess.run([self.eval_step,
                                                                 self.model.mean_loss,
                                                                 self.model.predictions,
                                                                 self.model.logits,
                                                                 self.model.input_gate],
                                                                feed_dict=feed_dict)

        target_indices = np.argmax(batch_chunk['answer'], axis=1)
        return [loss, predictions, target_indices, logits, batch_chunk['ids'],input_gate]

    def log_step_message(self, current_step, loss, acc, concept_loss, step_time, steps_in_epoch, is_train=True):
        log_fn = (is_train and log.info or log.infov)
        batch_size = self.model.batch_size
        log_fn((" [{split_mode:5} step {step:4d} / epoch {epoch:.2f}]  " +
                "batch total-loss: {total_loss:.5f}, concept-loss: {concept_loss:.5f}, accuracy: {acc:.5f} " +
                "({sec_per_batch:.3f} sec/batch, {instance_per_sec:.3f} instances/sec) | {train_tag}"
                ).format(split_mode=(is_train and 'train' or 'val'),
                         epoch=float(current_step)/steps_in_epoch,
                         step=current_step,
                         total_loss=loss, concept_loss=concept_loss,
                         acc=acc,
                         sec_per_batch=step_time,
                         instance_per_sec=batch_size / step_time,
                         train_tag=self.config.train_tag,
                         )
               )

    def evaluate(self, queue, dataset,generate_results,tag):
        log.info("Evaluate Phase")
        iter_length = int(len(dataset) / self.model.batch_size + 1)
        self.sess.run(self.eval_reset)

        total_input_gate = []
        for i in range(iter_length):
            loss, predictions, target_indices, logits, ids, input_gate = self.eval_single_step(queue)
            for j,key in enumerate(ids):
                total_input_gate.append((str(dataset.idx2word[predictions[j]]), input_gate[0][j],input_gate[1][j],input_gate[2][j],input_gate[3][j]))

            if i%100 == 0:
                target_word = dataset.idx2word[target_indices[0]]
                output_word = dataset.idx2word[predictions[0]]
                log.infov("[FIB {step:3d}/{total_length:3d}] target: {target}, prediction: {prediction}".format(
                    step=i, total_length=iter_length, target=target_word, prediction=output_word))
        total_acc = self.sess.run(self.accuracy)
        log.infov("[FIB] total accurycy: {acc:.5f}".format(acc=total_acc))


