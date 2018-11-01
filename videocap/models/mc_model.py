from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
import numpy as np
import time

from videocap.util import log
from videocap.datasets import data_util
import tensorflow.contrib.slim as slim
import tensorflow.contrib.rnn as rnn_cell

from videocap.utils import common_attention


_lstm_map = {
    'BasicLSTM': rnn_cell.BasicLSTMCell,
}


class MCGenerator(object):
    """
    """
    def __init__(self, config, word_embed):
        self.config = config
        self.batch_size = config.batch_size
        self.word_embed = word_embed
        self.vocab_size = word_embed.shape[0]
        self.name = 'RET_Generator'

        self.dropout_keep_prob = tf.placeholder_with_default(
            self.config.dropout_keep_prob, [])

        self.video_steps = config.video_steps
        self.initializer = tf.contrib.layers.xavier_initializer(
            uniform=False)
        self.video_cell = lambda:  _lstm_map[self.config.lstm_cell](
            num_units=self.config.hidden_dim, state_is_tuple=True)
        self.caption_cell = lambda: _lstm_map[self.config.lstm_cell](
            num_units=self.config.hidden_dim, state_is_tuple=True)

        if self.config.wav_data:
            self.channel_size = 2176
        else:
            self.channel_size = 2048
        if self.config.image_feature_net == 'vgg':
            self.channel_size=4096
        self.MULTICHOICE_COUNT = 5

    def get_feed_dict(self, batch_chunk):
        feed_dict = {
            self.video: batch_chunk['video_features'].astype(float),
            self.video_mask: batch_chunk['video_mask'].astype(float),
            self.mc_caption: batch_chunk['candidates'],
            self.mc_caption_mask: batch_chunk['candidates_mask'],
        }
        return feed_dict

    def get_placeholder(self):
        MULTICHOICE_COUNT = self.MULTICHOICE_COUNT
        video = tf.placeholder(tf.float32, [self.config.batch_size, self.config.video_steps, 1, 1, self.channel_size])
        video_mask = tf.placeholder(tf.float32, [self.config.batch_size, self.config.video_steps])
        mc_caption = tf.placeholder(tf.int32, [self.config.batch_size, MULTICHOICE_COUNT, self.config.caption_length])
        mc_caption_mask = tf.placeholder(tf.float32, [self.config.batch_size, MULTICHOICE_COUNT, self.config.caption_length])
        train_flag = tf.placeholder(tf.bool)

        result = {
            'video': video,
            'video_mask': video_mask,
            'mc_caption': mc_caption,
            'mc_caption_mask': mc_caption_mask,
            'train_flag': train_flag
        }
        return result

    def build_caption_embedding(self, input_seqs, name=None, reuse_variable=False):

        with tf.variable_scope("seq_embedding", reuse=reuse_variable), tf.device("/cpu:0"):
            seq_embeddings = tf.nn.embedding_lookup(self.word_embed_t, input_seqs, name=name)

        return seq_embeddings

    def build_video_embedding(self, video_cell, video, video_mask, reuse_variable):
        feat_type='cnn'
        if feat_type == 'cnn':
            with tf.variable_scope("video_cnn", reuse=reuse_variable) as scope:
                with slim.arg_scope([slim.conv2d],
                                    activation_fn=tf.nn.relu,
                                    padding='same',
                                    weights_initializer=self.initializer,
                                    weights_regularizer=slim.l2_regularizer(0.0005),
                                    normalizer_fn=slim.batch_norm,
                                    normalizer_params={'is_training':self.train_flag}):
                    'Already did dropout at front'

                    #video : [BxLx1x2048]
                    video_emb = tf.expand_dims(video, 2)
                    conv1 = slim.conv2d(video_emb, 2048, [3, 1], scope='conv1')
                    conv2 = slim.conv2d(conv1, 2048, [3, 1], scope='conv2')
                    conv3 = slim.conv2d(conv2, 2048, [3, 1],scope='conv3')
                    outputs = tf.reduce_mean(conv3, 2)

                    input_pass = outputs[:,:,0:1024]
                    input_gate = outputs[:,:,1024:]
                    input_gate = tf.sigmoid(input_gate)
                    outputs =  tf.multiply(input_pass, input_gate)
                    outputs = tf.concat([outputs,video],axis = 2)
        if feat_type == 'rnn':
            with tf.variable_scope("vid_rnn", reuse=reuse_variable) as scope:
                video_pool = []
                for i in range(self.config.caption_length):
                    video_pool.append(video[:, i, :])
                outputs, _, _ = tf.contrib.rnn.static_bidirectional_rnn(video_cell[0],
                                                                        video_cell[1],
                                                                        video_pool,
                                                                        dtype=tf.float32)
                outputs = tf.stack(outputs)
                outputs = tf.transpose(outputs, [1, 0, 2])
                outputs = tf.concat([outputs,video],2)

        outputs = slim.fully_connected(outputs, 512, scope='vid_fc', activation_fn=tf.nn.tanh, normalizer_fn=slim.batch_norm, normalizer_params={'is_training':self.train_flag})
        masked_outputs = outputs * tf.expand_dims(video_mask, 2)
        return masked_outputs

    def build_caption_encoder(self,
                              caption_cell,
                              reuse_variable=False):
        embedded_sentence = self.build_caption_embedding(self.caption_agg, name="embedding_sent")
        embedded_sentence = embedded_sentence * tf.expand_dims(self.caption_mask_agg, 2)
        self.emb_sent = tf.expand_dims(embedded_sentence,1)

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

        outputs = slim.fully_connected(outputs, 512, scope='cap_fc', activation_fn=tf.nn.tanh, normalizer_fn=slim.batch_norm, normalizer_params={'is_training':self.train_flag})

        rnn_output = outputs * tf.expand_dims(self.caption_mask_agg, 2)
        return rnn_output, embedded_sentence


    def fusion(self,
               vid_emb_state,
               cap_emb_state,
               raw_sent,
               reuse=False):
        '''
        vid_emb_state: [B, V, 512]
        cap_emb_state: [B, C, 512]
        raw_sent: [B, C, 300]
        '''
        states = []
        for i in range(int(self.config.video_steps)):
            vid_sample = tf.tile(tf.expand_dims(vid_emb_state[:,i,:],1),[1,self.config.caption_length,1])
            sum_repr = tf.multiply(vid_sample, cap_emb_state)
            #concat other infos
            states.append(sum_repr)
        # V x B x C x 512
        cnn_repr = tf.stack(states)
        # B x V x C x 512
        cnn_repr = tf.transpose(cnn_repr,[1,0,2,3])

        with slim.arg_scope([slim.fully_connected],
                            weights_regularizer=slim.l2_regularizer(0.0005),
                            activation_fn=tf.nn.tanh,
                            normalizer_fn=slim.batch_norm,
                            normalizer_params={'is_training':self.train_flag}):
            att_repr = cnn_repr
            h1 = slim.fully_connected(att_repr, 512, scope='rel_h1',activation_fn = tf.nn.tanh, reuse=reuse)
            input_gate1 = slim.fully_connected(h1,1,scope='rel_halp',activation_fn = tf.nn.sigmoid, reuse=reuse)
            h2 = slim.fully_connected(cnn_repr, 512, scope='rel_h2',activation_fn = tf.nn.tanh, reuse=reuse)
            h3 = slim.fully_connected(h2,512,scope='rel_h3',activation_fn = tf.nn.tanh, reuse=reuse)
        output1 = tf.multiply(h3,input_gate1)
        output1 = tf.multiply(tf.multiply(output1, tf.expand_dims(tf.expand_dims(self.video_mask_agg,2),3)), tf.expand_dims(tf.expand_dims(self.caption_mask_agg,1),3))

        with slim.arg_scope([slim.conv2d],
                            weights_initializer=self.initializer,
                            weights_regularizer=slim.l2_regularizer(0.0005),
                            reuse = reuse):

            conv1 = slim.conv2d(output1,256,[3,3],padding="Valid",scope='conv1',activation_fn=tf.nn.tanh)
            convalp1 = slim.conv2d(output1,1,[3,3],padding="Valid",scope='conv1alp',activation_fn = tf.nn.tanh)
            input_gate2 = tf.nn.sigmoid(convalp1)
            output2 = tf.multiply(conv1,input_gate2)
            output2 = tf.multiply(tf.multiply(output2, tf.expand_dims(tf.expand_dims(self.video_mask_list[0],2),3)), tf.expand_dims(tf.expand_dims(self.caption_mask_list[0],1),3))

            conv2 = slim.conv2d(output2,256,[3,3],padding="Valid",scope='conv2', activation_fn = tf.nn.tanh)
            convalp2 = slim.conv2d(output2,1,[3,3],padding="Valid",scope='conv2alp', activation_fn = tf.nn.tanh)
            input_gate3 = tf.nn.sigmoid(convalp2)
            output3 = tf.multiply(conv2,input_gate3)
            output3 = tf.multiply(tf.multiply(output3, tf.expand_dims(tf.expand_dims(self.video_mask_list[1],2),3)), tf.expand_dims(tf.expand_dims(self.caption_mask_list[1],1),3))

            conv3 = slim.conv2d(output3,256,[3,3],[2,2],padding="Valid",scope='conv3', activation_fn=tf.nn.tanh)
            convalp3 = slim.conv2d(output3,1,[3,3],[2,2],padding="Valid",scope='conv3alp', activation_fn = tf.nn.tanh)
            input_gate4 = tf.nn.sigmoid(convalp3)
            output4 = tf.multiply(conv3,input_gate4)
            output4 = tf.multiply(tf.multiply(output4, tf.expand_dims(tf.expand_dims(self.video_mask_list[2],2),3)), tf.expand_dims(tf.expand_dims(self.caption_mask_list[2],1),3))

        valid = tf.multiply(tf.reduce_sum(self.video_mask_list[2], axis=1), tf.reduce_sum(self.caption_mask_list[2],axis=1))
        sum_state = tf.div(tf.reduce_sum(output4, [1,2]), tf.expand_dims(valid,axis=1))

        self.input_gate = [input_gate1,input_gate2,input_gate3,input_gate4]
        return sum_state


    def build_model(self,
                    video,
                    video_mask,
                    mc_caption,
                    mc_caption_mask,
                    train_flag,
                    reuse_variable=False):


        self.video = video  # [batch_size, length, kernel, kernel, channel]
        self.video_mask = video_mask  # [batch_size, length]
        self.mc_caption = mc_caption  # [batch_size, length]
        self.mc_caption_mask = mc_caption_mask  # [batch_size, length]
        self.train_flag = train_flag

        self.word_embed_t = tf.Variable(self.word_embed, dtype=tf.float32, name="word_embed", trainable=True)
        #video drop
        self.squeezed_feat = tf.squeeze(self.video)
        self.embedded_feat = tf.reshape(self.squeezed_feat, [self.batch_size,
                                                             self.video_steps,
                                                             self.channel_size])
        #  [batch_size, length, channel_size]  Add pos embedding
        self.embedded_feat = common_attention.add_timing_signal_nd(self.embedded_feat)
        self.embedded_feat = self.embedded_feat * tf.expand_dims(video_mask, 2)

        #  [batch_size, length, channel_size]
        self.embedded_feat_drop = tf.nn.dropout(self.embedded_feat, self.dropout_keep_prob)

        MULTICHOICE_COUNT = self.MULTICHOICE_COUNT # e.g. 5
        batch_agg_index = np.concatenate(np.tile(np.arange(self.batch_size).reshape([self.batch_size, 1]),
                                         [1, MULTICHOICE_COUNT])) # [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, ...]

        self.video_agg      = tf.gather(self.embedded_feat_drop, batch_agg_index, name='video_agg')
        self.video_mask_agg = tf.gather(self.video_mask, batch_agg_index, name='video_mask_agg')
        video_mask_leng = tf.cast(tf.reduce_sum(self.video_mask_agg,1),tf.int32)
        self.caption_agg    = tf.reshape(self.mc_caption, [self.batch_size * MULTICHOICE_COUNT] + self.mc_caption.get_shape().as_list()[2:],
                                      name='caption_agg')
        self.caption_mask_agg = tf.reshape(self.mc_caption_mask, # [B x 5 x L] -> [5B x L]
                                           self.caption_agg.get_shape().as_list(), name='caption_mask_agg')
        caption_mask_leng = tf.cast(tf.reduce_sum(self.caption_mask_agg,1),tf.int32)

        self.agg_batch_size = self.batch_size * MULTICHOICE_COUNT

        #Mask list
        self.video_mask_list = []
        self.caption_mask_list = []
        max_len = self.config.caption_length
        for mi in range(2):
            video_mask_leng = tf.maximum(1, video_mask_leng-2)
            caption_mask_leng = tf.maximum(1, caption_mask_leng-2)
            max_len -= 2
            self.video_mask_list.append(tf.reverse(tf.sequence_mask(video_mask_leng,max_len,tf.float32),[-1]))
            self.caption_mask_list.append(tf.sequence_mask(caption_mask_leng,max_len,tf.float32))
        max_len = int((max_len-1)/2)
        video_mask_leng = tf.cast((video_mask_leng-1)/2,tf.int32)
        video_mask_leng = tf.maximum(1, video_mask_leng)
        caption_mask_leng = tf.cast((caption_mask_leng-1)/2,tf.int32)
        caption_mask_leng = tf.maximum(1, caption_mask_leng)
        self.video_mask_list.append(tf.reverse(tf.sequence_mask(video_mask_leng,max_len,tf.float32),[-1]))
        self.caption_mask_list.append(tf.sequence_mask(caption_mask_leng,max_len,tf.float32))


        assert self.agg_batch_size == self.video_agg.get_shape().as_list()[0]
        assert self.agg_batch_size == self.video_mask_agg.get_shape().as_list()[0]
        assert self.agg_batch_size == self.caption_agg.get_shape().as_list()[0]
        assert self.agg_batch_size == self.caption_mask_agg.get_shape().as_list()[0]

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
        self.video_cell = [self.video_cell1, self.video_cell2]

        self.caption_cell1 = rnn_cell.MultiRNNCell([self.caption_cell_d() for _ in range(self.config.num_layers)],
                                                state_is_tuple=True)
        self.caption_cell2 = rnn_cell.MultiRNNCell([self.caption_cell_d() for _ in range(self.config.num_layers)],
                                                state_is_tuple=True)
        self.caption_cell = [self.caption_cell1, self.caption_cell2]

        video_emb_state= self.build_video_embedding(self.video_cell,
                                              self.video_agg, self.video_mask_agg, reuse_variable)
        rnn_emb_state, raw_sent = self.build_caption_encoder(self.caption_cell, reuse_variable)

        with tf.variable_scope("multimodal", initializer=self.initializer) as scope:
            fuse = self.fusion(video_emb_state , rnn_emb_state,raw_sent, reuse=False)
            with slim.arg_scope([slim.fully_connected],
                                weights_regularizer=slim.l2_regularizer(0.0005),
                                normalizer_fn=slim.batch_norm,
                                normalizer_params={'is_training':self.train_flag}):
                logit = slim.fully_connected(fuse, 256, activation_fn=tf.nn.tanh, scope='fc1', normalizer_fn=slim.batch_norm, normalizer_params={'is_training':self.train_flag})
                logit = slim.fully_connected(logit, 256, activation_fn=tf.nn.tanh, scope='fc2', normalizer_fn=slim.batch_norm, normalizer_params={'is_training':self.train_flag})
                logit = slim.fully_connected(logit, 128, activation_fn=tf.nn.tanh, scope='fc3', normalizer_fn=slim.batch_norm, normalizer_params={'is_training':self.train_flag})
                logit = slim.fully_connected(logit, 1, activation_fn=None, scope='scorefn')
                score = logit

        batch_agg_index = np.concatenate(np.tile(np.arange(self.batch_size).reshape([self.batch_size, 1]),
                                        [1, MULTICHOICE_COUNT])) * MULTICHOICE_COUNT # [0, 0, 0, 0, 0, 5, 5, 5, 5, 1, ...]
        self.logit = -score
        self.sim = self.logit
        margin_loss = tf.maximum(0.0, 10.0 - tf.gather(self.sim, batch_agg_index, name='video_agg') + self.sim)
        margin_loss = tf.reduce_sum(margin_loss) / self.batch_size

        self.concept_loss =tf.constant(0)
        self.mean_loss = margin_loss

        with tf.variable_scope("accuracy"):
            # prediction tensor on test phase
            self.score = tf.reshape(self.logit, [self.batch_size, MULTICHOICE_COUNT])
            self.predictions = tf.argmax(
                self.score,
                dimension=1, name='argmax_predictions'
            )
            self.predictions.get_shape().assert_is_compatible_with([self.batch_size])

            self.correct_predictions = tf.cast(tf.equal(self.predictions, 0), tf.int32)
            self.acc = tf.reduce_mean(tf.cast(self.correct_predictions, "float"), name="accuracy")




class MCTrainer(object):

    def __init__(self, config, model, sess=None, train_summary_dir=None):
        self.sess = sess or tf.get_default_session()
        self.model = model
        self.config = config
        self.train_summary_dir = train_summary_dir
        self.MULTICHOICE_COUNT = self.model.MULTICHOICE_COUNT
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
            [step_op, self.model.mean_loss, self.model.acc,self.model.concept_loss, self.global_step,
             self.summary_mean_loss],
            feed_dict=feed_dict)
        if self.train_summary_writer is not None:
            self.train_summary_writer.add_summary(summary, current_step)
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
        _, loss, pred, logit, sim, input_gate = self.sess.run([self.eval_step,
                                                          self.model.mean_loss,
                                                          self.model.predictions,
                                                          self.model.logit,
                                                          self.model.sim,
                                                          self.model.input_gate],
                                                          feed_dict=feed_dict)
        return [loss, pred, logit, sim, input_gate]

    def log_step_message(self, current_step, loss, concept_loss, acc, step_time, steps_in_epoch, is_train=True):
        log_fn = (is_train and log.info or log.infov)
        batch_size = self.model.batch_size
        log_fn((" [{split_mode:5} step {step:4d} / epoch {epoch:.2f}]  " +
                "batch total-loss: {total_loss:.5f}, concept-loss: {concept_loss:.5f}, acc: {acc:.5f} " +
                "({sec_per_batch:.3f} sec/batch, {instance_per_sec:.3f} instances/sec) | {train_tag}"
                ).format(split_mode=(is_train and 'train' or 'val'),
                         epoch=float(current_step)/steps_in_epoch,
                         step=current_step,
                         total_loss=loss, concept_loss=concept_loss, acc=acc,
                         sec_per_batch=step_time,
                         instance_per_sec=batch_size / step_time,
                         train_tag=self.config.train_tag,
                         )
               )

    def evaluate(self, queue, dataset, global_step=None, generate_results=False, tag=''):
        log.info("Evaluate Phase")
        self.sess.run(self.eval_reset)

        iter_length = int(len(dataset) / self.model.batch_size + 1)
        for i in range(iter_length):
            loss, pred, logit, sim, input_gate = self.eval_single_step(queue)
            if i%50 == 0:
                log.infov("{}/{}".format(i, iter_length))
        total_acc = self.sess.run([self.accuracy])
        log.infov("[MC] total accuracy: {} ".format(total_acc) + tag)


