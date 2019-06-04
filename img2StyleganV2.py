import tensorflow as tf
import stylegan.dnnlib as dnnlib
import sys
sys.modules["dnnlib"] = dnnlib
from dnnlib.tflib.tfutil import convert_images_to_uint8
import stylegan.dnnlib.tflib as tflib
import stylegan.config as config
import pickle
import numpy as np
import PIL as PIL
import os
from  keras.preprocessing.image import load_img
from keras.applications.vgg16 import preprocess_input
import vgg
slim = tf.contrib.slim

# Tensorboard stuff
import time
experiment_dir = './experiments/' +  str(int(time.time()))
os.mkdir(experiment_dir)
def write_summary(value, tag, global_step, summary_writer):
    summary = tf.Summary()
    summary.value.add(tag=tag, simple_value=value)
    summary_writer.add_summary(summary, global_step)

summary_epoch = 500
num_styles = 6
dlatent_size = 512
lamda = 1
num_epochs = 0
learning_rate = 1e-2
intp_lamda = 0.5
syn_loss_lamda = 100

#----------------------------------------------------------------------------
# Convenience func that casts all of its arguments to tf.float32.

def fp32(*values):
    if len(values) == 1 and isinstance(values[0], tuple):
        values = values[0]
    values = tuple(tf.cast(v, tf.float32) for v in values)
    return values if len(values) >= 2 else values[0]

# Initialize TensorFlow.
tflib.init_tf()

# Load pre-trained network.
url = 'https://drive.google.com/uc?id=1MEGjdvVpUsu1jB4zrXZN7Y4kBBOzizDQ' # karras2019stylegan-ffhq-1024x1024.pkl
with dnnlib.util.open_url(url, cache_dir=config.cache_dir) as f:
    _G, D, Gs = pickle.load(f)

# get W
W_np = Gs.get_var('dlatent_avg')
W_np = np.reshape(np.tile(W_np, [2, num_styles]), [2, num_styles, dlatent_size])
W = tf.get_variable('optimal_dlatent', initializer=W_np)
tf.summary.histogram('optimal_dlatent', W)
# W_np = np.reshape(np.tile(W_np, [3]), [2, num_styles * 3, dlatent_size])

generated_img = Gs.components.synthesis.get_output_for(W, randomize_noise=False, structure='linear')
tf.summary.image('images/generated', convert_images_to_uint8(generated_img, nchw_to_nhwc=True))
generated_img = tf.transpose(generated_img, perm=[0, 2, 3, 1])
tf.summary.histogram('float_imgs/generated', generated_img)

W_synthesis = intp_lamda * W[0] + (1 - intp_lamda) * W[1]

# W_synthesis[:255] = W[0][:255]
W_synthesis = tf.expand_dims(W_synthesis, 0)

img_synthesis = Gs.components.synthesis.get_output_for(W_synthesis)
# fake_scores_out = fp32(D.get_output_for(img_synthesis, None))
img_synthesis = convert_images_to_uint8(img_synthesis, nchw_to_nhwc=True)
tf.summary.image('images/synthesis', img_synthesis)

real_img1 = load_img('andrew.png')
real_img2 = load_img('example.png')
real_img1 = np.expand_dims(real_img1, 0)
real_img2 = np.expand_dims(real_img2, 0)
real_img = np.vstack([real_img1, real_img2])
tf.summary.image('images/real', real_img)
real_img = preprocess_input(real_img, mode='tf')
tf.summary.histogram('float_imgs/real', real_img)
big_real_img = (real_img + 1) * (255 / 2)
tf.summary.histogram('int_imgs/real', big_real_img)
big_generated_img =  (generated_img + 1) * (255 / 2)
tf.summary.histogram('int_imgs/genereated', big_generated_img)
loss = lamda * tf.norm(big_real_img - big_generated_img)

# synthesis loss
# loss += syn_loss_lamda * (1 - fake_scores_out)

generated_img = tf.image.resize_images(generated_img, size=[224,224])
real_img = tf.image.resize_images(real_img, size=[224, 224])
_, end_points_generated = vgg.vgg_16(generated_img)
_, end_points_real = vgg.vgg_16(real_img)

layers = ['vgg_16/conv1/conv1_1', 'vgg_16/conv1/conv1_2', 'vgg_16/conv3/conv3_2', 'vgg_16/conv4/conv4_2']
for layer in layers:
    activation_generated = end_points_generated[layer]
    activation_real = end_points_real[layer]
    loss += tf.norm(activation_generated - activation_real)

optim = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = tf.contrib.layers.optimize_loss(
    loss, tf.train.get_global_step(), None, optim, summaries=['gradients'], variables=[W]
)

merge_summaries = tf.summary.merge_all()

with tf.get_default_session() as sess:
    # Create summary writter for tensorboard
    summary_writer = tf.summary.FileWriter(experiment_dir, sess.graph)

    restore = slim.assign_from_checkpoint_fn(
               'vgg_16.ckpt',
               slim.get_model_variables("vgg_16")
              )
    restore(sess)
    sess.run(W.initializer)
    sess.run(tf.variables_initializer(optim.variables()))

    _, summeries = sess.run([loss, merge_summaries])
    summary_writer.add_summary(summeries, 0)
    i = 0
    for i in range(num_epochs):
        input_feed = [train_op, loss]
        if i % summary_epoch == 0:
            input_feed.append(merge_summaries)
        outputs = sess.run(input_feed)
        loss_val = outputs[1]

        print("epoch " + str(i) + ": " + str(loss_val) + "\n")
        write_summary(loss_val, 'loss', i, summary_writer)
        if i % summary_epoch == 0:
            summary_writer.add_summary(outputs[2], i+1)

    # optimal_img = Gs.components.synthesis.get_output_for(W)
    tf.summary.histogram('output_img/float', optimal_img)
    optimal_img = convert_images_to_uint8(optimal_img, nchw_to_nhwc=True)
    tf.summary.histogram('output_img/int', optimal_img)
    optimal_img, summaries = sess.run([optimal_img, merge_summaries])
    summary_writer.add_summary(summaries, i)

    os.makedirs(config.result_dir, exist_ok=True)
    for i, img in enumerate(optimal_img):
        png_filename = os.path.join(config.result_dir, 'optimal{}.png'.format(i))
        PIL.Image.fromarray(optimal_img[0], 'RGB').save(png_filename)

    W_synthesis = intp_lamda * W[0] + (1 - intp_lamda) * W[1]
    W_synthesis = tf.expand_dims(W_synthesis, 0)
    # img_synthesis = Gs.components.synthesis.get_output_for(W_synthesis)
    img_synthesis = convert_images_to_uint8(img_synthesis, nchw_to_nhwc=True)
    img_synthesis = sess.run([img_synthesis])

    png_filename = os.path.join(config.result_dir, 'synthesis.png')
    PIL.Image.fromarray(img_synthesis[0], 'RGB').save(png_filename)
