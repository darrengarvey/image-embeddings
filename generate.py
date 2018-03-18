#!/usr/bin/env python

from __future__ import print_function

import os
import tensorflow as tf
import vgg16

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('path', '', 'Path of image file or directory containing images that will be  walked recursively')
tf.flags.DEFINE_string('output', 'embeddings.csv', 'Output file of embeddings')
tf.flags.DEFINE_integer('batch_size', 32, 'Batch size. Larger is faster but uses more memory')
tf.flags.DEFINE_integer('print_every', 100, 'Print progress every N iterations')


def read_files():
  if os.path.isdir(FLAGS.path):
    for root, _, files in os.walk(FLAGS.path):
      for f in files:
        filename = os.path.join(root, f)
        yield filename
  else:
    yield FLAGS.path

def parse_image(filename):
  image = tf.read_file(filename)
  # Force three channels so we can batch colour and greyscale images together
  image = tf.image.decode_jpeg(image, channels=3)
  image = tf.image.resize_images(image, [vgg16.DIMS[0], vgg16.DIMS[1]])
  return filename, image

def write_embedding(emb, outfile, label=None):
    if label:
        outfile.write(label.decode('utf-8') + ",")
    outfile.write(",".join([str(x) for x in emb]))
    outfile.write("\n")

def main(_):
  outdir = os.path.dirname(FLAGS.output)
  if not os.path.exists(outdir):
    os.makedirs(outdir)

  dataset = (tf.data.Dataset
                 .from_generator(read_files, (tf.string))
                 .map(parse_image)
                 .batch(FLAGS.batch_size))

  filenames, images = dataset.make_one_shot_iterator().get_next()
  model = vgg16.Vgg16(images)

  with tf.train.SingularMonitoredSession() as sess:
    with open(FLAGS.output, 'w') as outfile:
      i = 0
      while not sess.should_stop():
        filename, embedding = sess.run([filenames, model.embedding])
        for label, emb in zip(filename, embedding):
          write_embedding(emb, outfile, label)
        if i % FLAGS.print_every == 0:
          print('At iteration {}'.format(i))
        i += 1

if __name__ == '__main__':
  tf.app.run(main)
