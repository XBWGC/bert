# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Run BERT on SQuAD 1.1 and SQuAD 2.0."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import json
import math
import os
import random
import modeling
import six
import shutil
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import graph_util


flags = tf.flags
FLAGS = flags.FLAGS

# parameters
flags.DEFINE_string(
    "bert_config_file", 'models/bert_config.json',
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string("vocab_file", 'models/vocab.txt',
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string(
    "output_dir", 'saved_model',
    "The output directory where the model checkpoints will be written.")

flags.DEFINE_integer(
    "max_seq_length", 384,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_bool("do_predict", True, "Whether to run eval on the dev set.")

flags.DEFINE_integer("batch_size", 8,
                     "Total batch size for predictions.")

flags.DEFINE_bool("frozen_pb", False, "Whether save with frozen_pb or SavedModel")

flags.DEFINE_bool(
    "verbose_logging", False,
    "If true, all of the warnings related to data processing will be printed. "
    "A number of warnings are expected for a normal SQuAD evaluation.")

flags.DEFINE_bool(
    "version_2_with_negative", False,
    "If true, the SQuAD examples contain some that do not have an answer.")


def create_model(bert_config, is_training, input_ids, input_mask, segment_ids,
                 use_one_hot_embeddings):
  """Creates a classification model."""
  model = modeling.BertModel(
      config=bert_config,
      is_training=is_training,
      input_ids=input_ids,
      input_mask=input_mask,
      token_type_ids=segment_ids,
      use_one_hot_embeddings=use_one_hot_embeddings)

  final_hidden = model.get_sequence_output()

  final_hidden_shape = modeling.get_shape_list(final_hidden, expected_rank=3)
  batch_size = final_hidden_shape[0]
  seq_length = final_hidden_shape[1]
  hidden_size = final_hidden_shape[2]

  output_weights = tf.get_variable(
      "cls/squad/output_weights", [2, hidden_size],
      initializer=tf.truncated_normal_initializer(stddev=0.02))

  output_bias = tf.get_variable(
      "cls/squad/output_bias", [2], initializer=tf.zeros_initializer())

  final_hidden_matrix = tf.reshape(final_hidden,
                                   [batch_size * seq_length, hidden_size])
  logits = tf.matmul(final_hidden_matrix, output_weights, transpose_b=True)
  logits = tf.nn.bias_add(logits, output_bias)

  logits = tf.reshape(logits, [batch_size, seq_length, 2])
  logits = tf.transpose(logits, [2, 0, 1])

  unstacked_logits = tf.unstack(logits, axis=0)

  (start_logits, end_logits) = (unstacked_logits[0], unstacked_logits[1])

  return (start_logits, end_logits)


def model_fn_builder(bert_config, use_one_hot_embeddings, placeholders):
  """Returns `model_fn` closure."""

  input_ids = placeholders["input_ids"]
  input_mask = placeholders["input_mask"]
  segment_ids = placeholders["segment_ids"]

  (start_logits, end_logits) = create_model(
      bert_config=bert_config,
      is_training=False,
      input_ids=input_ids,
      input_mask=input_mask,
      segment_ids=segment_ids,
      use_one_hot_embeddings=use_one_hot_embeddings)

  predictions = {
      "start_logits": start_logits,
      "end_logits": end_logits,
  }

  return predictions



def get_feed_dict(placeholders, shape):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""
  np.random.seed(12345)

  feed_dict = {}
  feed_dict[placeholders['input_ids']] = np.random.randint(0, 30522, shape, dtype=np.int32)
  feed_dict[placeholders['input_mask']] = np.random.randint(0, 2, shape, dtype=np.int32)
  feed_dict[placeholders['segment_ids']] = np.random.randint(0, 2, shape, dtype=np.int32)

  return feed_dict


def main(_):
  if os.path.exists(FLAGS.output_dir):
    shutil.rmtree(FLAGS.output_dir)
  tf.gfile.MakeDirs(FLAGS.output_dir)

  tf.logging.set_verbosity(tf.logging.INFO)

  bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

  placeholders = {}
  shape = (FLAGS.batch_size, FLAGS.max_seq_length)
  placeholders["input_ids"] = tf.placeholder(tf.int32, shape, name='input_ids')
  placeholders["input_mask"] = tf.placeholder(tf.int32, shape, name='input_mask')
  placeholders["segment_ids"] = tf.placeholder(tf.int32, shape, name='segment_ids')
  predictions = model_fn_builder(
      bert_config=bert_config,
      use_one_hot_embeddings=False,
      placeholders=placeholders)

  feed_dict = get_feed_dict(placeholders, shape)
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    output = sess.run(predictions, feed_dict)
    for key in output.keys():
      print(output[key].shape)

    if FLAGS.frozen_pb:
      output = [predictions['start_logits'], predictions['end_logits']]
      graph_def = graph_util.convert_variables_to_constants(sess, sess.graph_def, output)
      with tf.gfile.FastGFile(FLAGS.output_dir + '/bert_base.pb', mode='wb') as f:
        f.write(graph_def.SerializeToString())
    else:
      print("Exporting model to ", FLAGS.output_dir)
      builder = tf.saved_model.builder.SavedModelBuilder(FLAGS.output_dir)

      ids  = tf.saved_model.utils.build_tensor_info(placeholders["input_ids"])
      mask = tf.saved_model.utils.build_tensor_info(placeholders["input_mask"])
      seg  = tf.saved_model.utils.build_tensor_info(placeholders["segment_ids"])

      start = tf.saved_model.utils.build_tensor_info(predictions["start_logits"])
      end   = tf.saved_model.utils.build_tensor_info(predictions["end_logits"])

      prediction_signature = (
          tf.saved_model.signature_def_utils.build_signature_def(
              inputs={'input_ids': ids, 'input_mask': mask, 'segment_ids': seg},
              outputs={'start_logits': start, 'end_logits': end},
              method_name=tf.saved_model.signature_constants
              .PREDICT_METHOD_NAME))

      builder.add_meta_graph_and_variables(
          sess, [tf.saved_model.tag_constants.SERVING],
          signature_def_map={
              'bert_base':
                  prediction_signature,
              tf.saved_model.signature_constants
              .DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                  prediction_signature,
          },
          main_op=tf.tables_initializer(),
          strip_default_attrs=True)
      builder.save()

      print('Done exporting!')


if __name__ == "__main__":
  flags.mark_flag_as_required("vocab_file")
  flags.mark_flag_as_required("bert_config_file")
  flags.mark_flag_as_required("output_dir")
  tf.app.run()
