import time
import os
import pprint
import tensorflow as tf
from videocap.datasets.lsmdc import DatasetLSMDC
from videocap.datasets import data_util
from videocap.util import log

from videocap.models.fib_model import FIBGenerator, FIBTrainer
from videocap.models.mc_model import MCGenerator, MCTrainer

from videocap.datasets.batch_queue import BatchQueue
from videocap.configuration import ModelConfig, TrainConfig
import json

# For debug purporses
from tqdm import tqdm
import hickle as hkl
import numpy as np

pp = pprint.PrettyPrinter(indent=2)

tf.flags.DEFINE_string("tag","","Tag for test")
tf.flags.DEFINE_string("checkpoint","","Checkpoint Dir")
# Print parameters
FLAGS = tf.flags.FLAGS


MODELS = {
    'FIB': FIBGenerator,
    'MC' : MCGenerator,
}
MODEL_TRAINERS = {
    'FIB': FIBTrainer,
    'MC' : MCTrainer,
}

def main(argv):
    model_config = ModelConfig()
    train_config = TrainConfig()

    base_dir =  os.path.join("checkpoint",train_config.train_tag+"_"+FLAGS.tag)
    checkpoint_dir = os.path.join(base_dir,"model.ckpt")
    logits_dir = os.path.join(base_dir,"logits_")
    if not os.path.exists(base_dir):
        os.mkdir(base_dir)
    #
    train_dataset = DatasetLSMDC(dataset_name='train',
                                 image_feature_net=model_config.image_feature_net,
                                 layer=model_config.layer,
                                 max_length=model_config.caption_length,
                                 max_vid_length=model_config.video_steps,
                                 max_n_videos=None,
                                 data_type=train_config.train_tag,
                                 attr_length=model_config.attr_length,
                                 wav_data = model_config.wav_data)
    validation_dataset = DatasetLSMDC(dataset_name='test',
                                      image_feature_net=model_config.image_feature_net,
                                      layer=model_config.layer,
                                      max_length=model_config.caption_length,
                                      max_vid_length = model_config.video_steps,
                                      max_n_videos=None,
                                      data_type=train_config.train_tag,
                                      attr_length=model_config.attr_length,
                                      wav_data = model_config.wav_data)
    train_dataset.build_word_vocabulary()
    validation_dataset.share_word_vocabulary_from(train_dataset)
    train_iter = train_dataset.batch_iter(train_config.num_epochs, model_config.batch_size)
    train_queue = BatchQueue(train_iter, name='train')
    val_iter = validation_dataset.batch_iter(20*train_config.num_epochs, model_config.batch_size, shuffle=False)
    val_queue = BatchQueue(val_iter, name='test')
    train_queue.start_threads()
    val_queue.start_threads()

    g = tf.Graph()
    with g.as_default():
        global session, model, trainer
        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        session = tf.Session(graph=g, config=tf_config)

        model = MODELS[train_config.train_tag](model_config, train_dataset.word_matrix)

        log.info("Build the model...")
        model.build_model(**model.get_placeholder())
        trainer = MODEL_TRAINERS[train_config.train_tag](train_config, model, session)


        steps_in_epoch = int(np.ceil(len(train_dataset) / model.batch_size))

        saver = tf.train.Saver(max_to_keep=10)


        if train_config.load_from_ckpt is not None:
            log.info("Restoring parameter from {}".format(train_config.load_from_ckpt))
            session.run(tf.global_variables_initializer())
            saver.restore(session, train_config.load_from_ckpt)
        else:
            session.run(tf.global_variables_initializer())
        for step in range(train_config.max_steps):
            step_result = trainer.run_single_step(
                 queue=train_queue, is_train=True)

            if step_result['current_step'] % train_config.steps_per_logging == 0:
                step_result['steps_in_epoch'] = steps_in_epoch
                trainer.log_step_message(**step_result)

            if step_result['current_step'] % train_config.steps_per_evaluate == 0 or train_config.print_evaluate:
                trainer.evaluate(queue=val_queue, dataset=validation_dataset, generate_results=True, tag=FLAGS.tag)

                print("SAVE MODEL"+FLAGS.tag)
                saver.save(session, checkpoint_dir, global_step=step)

        train_queue.thread_close()
        val_queue.thread_close()

if __name__ == '__main__':
    tf.app.run(main=main)
