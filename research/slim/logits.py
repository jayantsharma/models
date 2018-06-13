import tensorflow as tf
import numpy as np
import os, glob, sys
import pickle
from imageio import imread, imwrite
from scipy.misc import imresize
from scipy.io import savemat

from nets import nets_factory
from preprocessing import preprocessing_factory
from datasets.patches import _NUM_CLASSES


MODEL_NAME = 'inception_v3'
CHECKPOINT_DIR = '/home/jayant/features_filtered/augmented_conf_optflow_1/model'
BATCH_SIZE = 64
# DATASET_DIR = '/home/jayant/TraderJoe/image'


if __name__ == '__main__':
    DATASET_DIR = sys.argv[1]
    START_NUM = int(sys.argv[2])
    END_NUM = int(sys.argv[3])

    ### DEFINE GRAPH ###
    image_slices = tf.placeholder(tf.uint8, shape=[None,100,100,3])
    dataset = tf.data.Dataset.from_tensor_slices(image_slices)

    # Select the model #
    network_fn = nets_factory.get_network_fn(MODEL_NAME, num_classes=_NUM_CLASSES, is_training=False)

    # Preprocessing #
    image_preprocessing_fn = preprocessing_factory.get_preprocessing(MODEL_NAME, is_training=False)
    eval_image_size = network_fn.default_image_size
    # image = image_preprocessing_fn(image, eval_image_size, eval_image_size)
    def _preprocess(image):
        return image_preprocessing_fn(image, eval_image_size, eval_image_size)
    dataset = dataset.map(_preprocess)
    batched_dataset = dataset.batch(BATCH_SIZE)
    iterator = batched_dataset.make_initializable_iterator()
    images = iterator.get_next()

    # Inference #
    logits, _ = network_fn(images)
    probs = tf.nn.softmax(logits)
    labels = tf.argmax(logits, 1)

    sess = tf.Session()
    saver = tf.train.Saver()
    saver.restore(sess, tf.train.latest_checkpoint(CHECKPOINT_DIR))


    # Load labels
    with open('/home/jayant/features_filtered/augmented/labels.txt') as f:
        lines = f.readlines()
        label_map = np.array([line.strip().split(':')[1] for line in lines])

    # imgpaths = glob.glob('{}/*.jpg'.format(DATASET_DIR))
    # img_slices = [ imread(imgpath).astype('float32') for imgpath in sorted(imgpaths) ]
    # imgpaths = imgpaths[:1]
    # for imgpath in imgpaths:

    for n in range(START_NUM, END_NUM):
        if n % 50 == 0:
            print('------------ IMAGE {} -------------'.format(n), flush=True)
        imgpath = '{}/image{:07d}.jpg'.format(DATASET_DIR, n)
        img = imread(imgpath)
        img_slices = []
        for hstart in range(0,700,100):
            hend = hstart+100
            for wstart in range(0,1280,100):
                wend = min(wstart+100,1280)
                img_slice = img[hstart:hend,wstart:wend]
                if wend-wstart < 100:
                    img_slice = imresize(img_slice, (100,100), interp='bilinear')
                img_slices.append(np.expand_dims(img_slice,axis=0))

        img_slices = np.concatenate(img_slices)
        sess.run(iterator.initializer, feed_dict={image_slices: img_slices})

        res_probs, res_labels = [], []
        while True:
            try:
                # res.append(sess.run(labels))
                lg, ps, ll = sess.run([logits, probs, labels])
                res_probs.append(ps)
                res_labels.append(ll)
            except tf.errors.OutOfRangeError:
                break
        res_probs = np.concatenate(res_probs)
        res_labels = np.concatenate(res_labels)
        # print(res)
        # res = label_map[res]
        fidx = imgpath.split('/')[-1].split('.')[0][5:]
        pickle.dump((res_probs, res_labels), open('{}/labels_{}.pkl'.format(DATASET_DIR, fidx), 'wb'))
        savemat(open('{}/labels_{}.mat'.format(DATASET_DIR, fidx), 'wb'), { 'probs': res_probs, 'labels': res_labels })
