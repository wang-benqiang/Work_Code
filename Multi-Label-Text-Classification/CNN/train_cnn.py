# -*- coding:utf-8 -*-
__author__ = 'Wang'

import os
import sys
import time
import logging
import numpy as np
import tensorflow as tf
from os.path import dirname, abspath
path =dirname(dirname(abspath(__file__)))
sys.path.append(path)
from tensorboard.plugins import projector
from text_cnn import TextCNN
from data_utils import checkmate as cm
from data_utils import data_helpers as dh
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score




# Data Parameters
tf.flags.DEFINE_string("training_data_file", '/hhdxx3/wang.benqiang/Multi-Label-Text-Classification-word/data/tfrecord/train.tfrecord', "Data source for the training data.")
tf.flags.DEFINE_string("validation_data_file", '/hhdxx3/wang.benqiang/Multi-Label-Text-Classification-word/data/tfrecord/valid.tfrecord', "Data source for the validation data.")
tf.flags.DEFINE_string("train_or_restore", 'T', "Train or Restore.")

# Model Hyperparameters
tf.flags.DEFINE_float("learning_rate", 0.001, "The learning rate (default: 0.001)")
tf.flags.DEFINE_integer("pad_seq_len", 30, "Recommended padding Sequence length of data (depends on the data)")
tf.flags.DEFINE_integer("embedding_dim", 300, "Dimensionality of character embedding (default: 300)")
tf.flags.DEFINE_integer("embedding_type", 1, "The embedding type (default: 1)")
tf.flags.DEFINE_integer("fc_hidden_size", 1024, "Hidden size for fully connected layer (default: 1024)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")
tf.flags.DEFINE_integer("num_classes", 4848, "Number of labels (depends on the task)")
tf.flags.DEFINE_integer("top_num", 3, "Number of top K prediction classes (default: 3)")
tf.flags.DEFINE_float("threshold", 0.05, "Threshold for prediction classes (default: 0.5)")
tf.flags.DEFINE_integer("train_data_num",2035384,"Number of training data")
tf.flags.DEFINE_integer("valid_data_num",7951,"Number of validation data")


# Training Parameters
tf.flags.DEFINE_integer("batch_size", 256, "Batch Size (default: 256)")
tf.flags.DEFINE_integer("num_epochs", 100, "Number of training epochs (default: 100)")
tf.flags.DEFINE_integer("evaluate_every", 200, "Evaluate model on dev set after this many steps (default: 5000)")
tf.flags.DEFINE_float("norm_ratio", 2, "The ratio of the sum of gradients norms of trainable variable (default: 1.25)")
tf.flags.DEFINE_integer("decay_steps", 500, "how many steps before decay learning rate. (default: 500)")
tf.flags.DEFINE_float("decay_rate", 0.95, "Rate of decay for learning rate. (default: 0.95)")
tf.flags.DEFINE_integer("checkpoint_every", 500, "Save model after this many steps (default: 1000)")
tf.flags.DEFINE_integer("num_checkpoints", 1, "Number of checkpoints to store (default: 50)")
# tf.flags.DEFINE_string("tfrecord_dir_path", '../data/tfrecord', "tfrecord_dir_path")


# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
tf.flags.DEFINE_boolean("gpu_options_allow_growth", True, "Allow gpu options growth")

FLAGS = tf.flags.FLAGS
FLAGS(sys.argv)

if FLAGS.train_or_restore == 'T':
    # logger = dh.logger_fn(r"tflog", r"logs/my_training-{0}.log".format(time.asctime()))
    logger = dh.logger_fn(r"tflog", r"logs/my_training.log")
if FLAGS.train_or_restore == 'R':
    logger = dh.logger_fn(r"tflog", r"logs/my_restore.log")

dilim = '-' * 100
logger.info('\n'.join([dilim, *['{0:>50}|{1:<50}'.format(attr.upper(), FLAGS.__getattr__(attr))
                                for attr in sorted(FLAGS.__dict__['__wrapped'])], dilim]))

def train_cnn():
    """Training CNN model."""

    # Load sentences, labels, and trainin


    pretrained_word2vec_matrix=np.load('/hhdxx3/wang.benqiang/Multi-Label-Text-Classification-word/data/word_vec/new_word_vector.npy')

    # Build a graph and cnn object
    with tf.Graph().as_default():
        #loading training data and validation data
        _, train_batch_features_content, train_batch_labels_index = dh.load_tfrecord(
            FLAGS.training_data_file, FLAGS.batch_size, FLAGS.pad_seq_len,FLAGS.num_classes)
        _, valid_batch_features_content, valid_batch_labels_index = dh.load_tfrecord(
            FLAGS.validation_data_file, FLAGS.batch_size, FLAGS.pad_seq_len, FLAGS.num_classes)

        #GPU configuration
        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)
        session_conf.gpu_options.allow_growth = FLAGS.gpu_options_allow_growth
        session_conf.gpu_options.per_process_gpu_memory_fraction=0.9
        sess = tf.Session(config=session_conf)

        #tfrecord configuration
        coor = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coor)

        with sess.as_default():
            cnn = TextCNN(
                sequence_length=FLAGS.pad_seq_len,
                num_classes=FLAGS.num_classes,
                fc_hidden_size=FLAGS.fc_hidden_size,
                embedding_size=FLAGS.embedding_dim,
                embedding_type=FLAGS.embedding_type,
                filter_sizes=list(map(int, FLAGS.filter_sizes.split(','))),
                num_filters=FLAGS.num_filters,
                l2_reg_lambda=FLAGS.l2_reg_lambda,
                pretrained_embedding=pretrained_word2vec_matrix)
            # Define training procedure
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                learning_rate = tf.train.exponential_decay(learning_rate=FLAGS.learning_rate,
                                                           global_step=cnn.global_step, decay_steps=FLAGS.decay_steps,
                                                           decay_rate=FLAGS.decay_rate, staircase=True)
                optimizer = tf.train.AdamOptimizer(learning_rate)
                grads, vars = zip(*optimizer.compute_gradients(cnn.loss))
                grads, _ = tf.clip_by_global_norm(grads, clip_norm=FLAGS.norm_ratio)
                train_op = optimizer.apply_gradients(zip(grads, vars), global_step=cnn.global_step, name="train_op")
            # Keep track of gradient values and sparsity (optional)
            grad_summaries = []
            for g, v in zip(grads, vars):
                if g is not None:
                    grad_hist_summary = tf.summary.histogram("{0}/grad/hist".format(v.name), g)
                    sparsity_summary = tf.summary.scalar("{0}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                    grad_summaries.append(grad_hist_summary)
                    grad_summaries.append(sparsity_summary)
            grad_summaries_merged = tf.summary.merge(grad_summaries)


            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs1"))
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            best_checkpoint_dir = os.path.abspath(os.path.join(out_dir, "bestcheckpoints"))

            # Summaries for loss
            loss_summary = tf.summary.scalar("loss", cnn.loss)

            # Train summaries
            train_summary_op = tf.summary.merge([loss_summary, grad_summaries_merged])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            # Validation summaries
            validation_summary_op = tf.summary.merge([loss_summary])
            validation_summary_dir = os.path.join(out_dir, "summaries", "validation")
            validation_summary_writer = tf.summary.FileWriter(validation_summary_dir, sess.graph)

            saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)
            best_saver = cm.BestCheckpointSaver(save_dir=best_checkpoint_dir, num_to_keep=1, maximize=True)

            if FLAGS.train_or_restore == 'R':
                # Load cnn model
                logger.info("✔︎ Loading model...")
                checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)
                logger.info(checkpoint_file)

                # Load the saved meta graph and restore variables
                saver = tf.train.import_meta_graph("{0}.meta".format(checkpoint_file))
                saver.restore(sess, checkpoint_file)
            else:
                # pass
                if not os.path.exists(checkpoint_dir):
                    os.makedirs(checkpoint_dir)
                sess.run(tf.global_variables_initializer())
                sess.run(tf.local_variables_initializer())
                # Embedding visualization config
                config = projector.ProjectorConfig()
                embedding_conf = config.embeddings.add()
                embedding_conf.tensor_name = "embedding"
                # embedding_conf.metadata_path = FLAGS.metadata_file
                #
                projector.visualize_embeddings(train_summary_writer, config)
                projector.visualize_embeddings(validation_summary_writer, config)

                # Save the embedding visualization
                saver.save(sess, os.path.join(out_dir, "embedding", "embedding.ckpt"))

            current_step = sess.run(cnn.global_step)
            def train_step(train_batch_features_content,train_batch_labels_index):
                """A single training step"""
                #print(1)
                feed_dict = {
                    cnn.input_x:train_batch_features_content,
                    cnn.input_y:train_batch_labels_index,
                    cnn.dropout_keep_prob: FLAGS.dropout_keep_prob,
                    cnn.is_training: True
                }
                _, step, summaries, loss = sess.run(
                    [train_op, cnn.global_step, train_summary_op, cnn.loss], feed_dict)
                #print(1)
                logger.info("step {0}: loss {1:g}".format(step, loss))
                train_summary_writer.add_summary(summaries, step)

            def validation_step(num_valid_steps, writer=None):
                """Evaluates model on a validation set"""
                # batches_validation = dh.batch_iter(list(zip(x_val, y_val)), FLAGS.batch_size, 1)

                # Predict classes by threshold or topk ('ts': threshold; 'tk': topk)
                eval_counter, eval_loss = 0, 0.0

                eval_pre_tk = [0.0] * FLAGS.top_num
                eval_rec_tk = [0.0] * FLAGS.top_num
                eval_F_tk = [0.0] * FLAGS.top_num

                true_onehot_labels = []
                predicted_onehot_scores = []
                predicted_onehot_labels_ts = []
                predicted_onehot_labels_tk = [[] for _ in range(FLAGS.top_num)]

                for i in range(num_valid_steps):
                    # x_batch_val, y_batch_val = zip(*batch_validation)
                    x_batch_val,y_batch_val=sess.run([valid_batch_features_content,valid_batch_labels_index])
                    feed_dict = {
                        cnn.input_x: x_batch_val,
                        cnn.input_y: y_batch_val,
                        cnn.dropout_keep_prob: 1.0,
                        cnn.is_training: False
                    }
                    step, summaries, scores, cur_loss = sess.run(
                        [cnn.global_step, validation_summary_op, cnn.scores, cnn.loss], feed_dict)

                    # Prepare for calculating metrics
                    for i in y_batch_val:
                        true_onehot_labels.append(i)
                    for j in scores:
                        predicted_onehot_scores.append(j)

                    # Predict by threshold
                    batch_predicted_onehot_labels_ts = \
                        dh.get_onehot_label_threshold(scores=scores, threshold=FLAGS.threshold)

                    for k in batch_predicted_onehot_labels_ts:
                        predicted_onehot_labels_ts.append(k)

                    # Predict by topK
                    for top_num in range(FLAGS.top_num):
                        batch_predicted_onehot_labels_tk = dh.get_onehot_label_topk(scores=scores, top_num=top_num+1)

                        for i in batch_predicted_onehot_labels_tk:
                            predicted_onehot_labels_tk[top_num].append(i)

                    eval_loss = eval_loss + cur_loss
                    eval_counter = eval_counter + 1

                    if writer:
                        writer.add_summary(summaries, step)

                eval_loss = float(eval_loss / eval_counter)

                # Calculate Precision & Recall & F1 (threshold & topK)
                eval_pre_ts = precision_score(y_true=np.array(true_onehot_labels),
                                              y_pred=np.array(predicted_onehot_labels_ts), average='micro')
                eval_rec_ts = recall_score(y_true=np.array(true_onehot_labels),
                                           y_pred=np.array(predicted_onehot_labels_ts), average='micro')
                eval_F_ts = f1_score(y_true=np.array(true_onehot_labels),
                                     y_pred=np.array(predicted_onehot_labels_ts), average='micro')

                for top_num in range(FLAGS.top_num):
                    eval_pre_tk[top_num] = precision_score(y_true=np.array(true_onehot_labels),
                                                           y_pred=np.array(predicted_onehot_labels_tk[top_num]),
                                                           average='micro')
                    eval_rec_tk[top_num] = recall_score(y_true=np.array(true_onehot_labels),
                                                        y_pred=np.array(predicted_onehot_labels_tk[top_num]),
                                                        average='micro')
                    eval_F_tk[top_num] = f1_score(y_true=np.array(true_onehot_labels),
                                                  y_pred=np.array(predicted_onehot_labels_tk[top_num]),
                                                  average='micro')

                # Calculate the average AUC
                eval_auc = roc_auc_score(y_true=np.array(true_onehot_labels),
                                         y_score=np.array(predicted_onehot_scores), average='micro')
                # Calculate the average PR
                eval_prc = average_precision_score(y_true=np.array(true_onehot_labels),
                                                   y_score=np.array(predicted_onehot_scores), average='micro')

                return eval_loss, eval_auc, eval_prc, eval_rec_ts, eval_pre_ts, eval_F_ts, \
                       eval_rec_tk, eval_pre_tk, eval_F_tk


            # Training loop. For each batch...
            num_train_steps=int(FLAGS.train_data_num / FLAGS.batch_size * FLAGS.num_epochs)
            num_valid_steps=int(FLAGS.valid_data_num / FLAGS.batch_size )+1
            for i in range(num_train_steps):
               # logger.info('hahaha')
                train_batch_features_content_,train_batch_labels_index_=sess.run([train_batch_features_content,train_batch_labels_index])
                print(train_batch_features_content_.shape)
                train_step(train_batch_features_content_,train_batch_labels_index_)
                current_step = tf.train.global_step(sess, cnn.global_step)

                if current_step % FLAGS.evaluate_every == 0:
                    logger.info("\nEvaluation:")
                    # valid_batch_features_content, valid_batch_labels_index = sess.run([valid_batch_features_content, valid_batch_labels_index])
                    # eval_loss, eval_auc, eval_prc, \
                    # eval_rec_ts, eval_pre_ts, eval_F_ts, eval_rec_tk, eval_pre_tk, eval_F_tk = \
                    #     validation_step(valid_batch_features_content, valid_batch_labels_index, writer=validation_summary_writer)
                    eval_loss, eval_auc, eval_prc, \
                    eval_rec_ts, eval_pre_ts, eval_F_ts, eval_rec_tk, eval_pre_tk, eval_F_tk = \
                        validation_step(num_valid_steps,writer=validation_summary_writer)



                    logger.info("All Validation set: Loss {0:g} | AUC {1:g} | AUPRC {2:g}"
                                .format(eval_loss, eval_auc, eval_prc))

                    # Predict by threshold
                    logger.info("☛ Predict by threshold: Precision {0:g}, Recall {1:g}, F {2:g}"
                                .format(eval_pre_ts, eval_rec_ts, eval_F_ts))

                    # Predict by topK
                    logger.info("☛ Predict by topK:")
                    for top_num in range(FLAGS.top_num):
                        logger.info("Top{0}: Precision {1:g}, Recall {2:g}, F {3:g}"
                                    .format(top_num+1, eval_pre_tk[top_num], eval_rec_tk[top_num], eval_F_tk[top_num]))
                    best_saver.handle(eval_prc, sess, current_step)
                if current_step % FLAGS.checkpoint_every == 0:
                    checkpoint_prefix = os.path.join(checkpoint_dir, "model")
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    logger.info("✔︎ Saved model checkpoint to {0}\n".format(path))
                if current_step % FLAGS.train_data_num == 0:
                    current_epoch = current_step // FLAGS.train_data_num
                    logger.info("✔︎ Epoch {0} has finished!".format(current_epoch))
        coor.request_stop()
        coor.join(threads)
    logger.info("✔︎ Done.")





if __name__ == '__main__':
    train_cnn()
