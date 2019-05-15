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
summary_epoch = 1

num_styles = 18
dlatent_size = 512
lamda = 1
num_epochs = 0
learning_rate = 1e-6

# Initialize TensorFlow.
tflib.init_tf()

# Load pre-trained network.
url = 'https://drive.google.com/uc?id=1MEGjdvVpUsu1jB4zrXZN7Y4kBBOzizDQ' # karras2019stylegan-ffhq-1024x1024.pkl
with dnnlib.util.open_url(url, cache_dir=config.cache_dir) as f:
    _G, _D, Gs = pickle.load(f)

# Print network details.
Gs.print_layers()

W_np = Gs.get_var('dlatent_avg')
W_np = np.reshape(np.tile(W_np, [num_styles]), [1, dlatent_size, num_styles])
W_np = np.transpose(W_np, [0, 2, 1])
W = tf.get_variable('optimal_dlatent', initializer=W_np)
tf.summary.histogram('optimal_dlatent', W)
# Z = np.random.randn(1, 512)
# W = Gs.components.mapping.get_output_for(tf.get_variable('Z', initializer=Z), None)
# Apply truncation trick.
# with tf.variable_scope('Truncation'):
#     layer_idx = np.arange(num_styles)[np.newaxis, :, np.newaxis]
#     ones = np.ones(layer_idx.shape, dtype=np.float32)
#     coefs = tf.where(layer_idx < 8, 0.7 * ones, ones)
#     W = tflib.lerp(Gs.get_var('dlatent_avg'), W, coefs)
# tf.summary.histogram('W', W)
generated_img = Gs.components.synthesis.get_output_for(W, randomize_noise=False, structure='linear', is_validation=True)
generated_img = tf.transpose(generated_img, perm=[0, 2, 3, 1])
tf.summary.histogram('float_imgs/generated', generated_img)

real_img = load_img('example.png')
real_img = np.expand_dims(real_img, 0)
real_img = preprocess_input(real_img, mode='tf')
tf.summary.histogram('float_imgs/real', real_img)
real_img = (real_img + 1) * (255 / 2)
tf.summary.histogram('int_imgs/real', real_img)
generated_img =  (generated_img + 1) * (255 / 2)
tf.summary.histogram('int_imgs/genereated', generated_img)
loss = lamda * tf.norm(real_img - generated_img)

# generated_img = tf.image.resize_images(generated_img, size=[224,224])
# real_img = tf.image.resize_images(real_img, size=[224, 224])
# _, end_points_real = vgg.vgg_16(real_img)
# _, end_points_generated = vgg.vgg_16(generated_img)

# loss = 0
# layers = ['vgg_16/conv1/conv1_1', 'vgg_16/conv1/conv1_2', 'vgg_16/conv3/conv3_2', 'vgg_16/conv4/conv4_2']
# for layer in layers:
#     activation_generated = end_points_generated[layer]
#     activation_real = end_points_real[layer]
#     loss += tf.norm(activation_generated - activation_real)

# optim = tf.train.AdamOptimizer(learning_rate=0.01)
# train_op = optim.minimize(loss, var_list=[W])
train_op = tf.contrib.layers.optimize_loss(
    loss, tf.train.get_global_step(), learning_rate, 'SGD', summaries=['gradients'], variables=[W]
)

merge_summaries = tf.summary.merge_all()

with tf.Session() as sess:
    # Create summary writter for tensorboard
    summary_writer = tf.summary.FileWriter(experiment_dir, sess.graph)

    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    # restore = slim.assign_from_checkpoint_fn(
    #            'vgg_16.ckpt',
    #            slim.get_model_variables("vgg_16")
    #           )
    sess.run(init_op)
    # restore(sess)
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

    # optimal_img = Gs.components.synthesis.get_output_for(W, use_noise=False, randomize_noise=False, blur_filter=None)
    optimal_img = Gs.components.synthesis.get_output_for(W, use_noise=False)
    tf.summary.histogram('output_img/float', optimal_img)
    optimal_img = convert_images_to_uint8(optimal_img, nchw_to_nhwc=True)
    tf.summary.histogram('output_img/int', optimal_img)
    optimal_img, summaries = sess.run([optimal_img, merge_summaries])
    print(np.max(optimal_img), np.min(optimal_img))
    summary_writer.add_summary(summaries, i)

    os.makedirs(config.result_dir, exist_ok=True)
    png_filename = os.path.join(config.result_dir, 'optimal.png')
    PIL.Image.fromarray(optimal_img[0], 'RGB').save(png_filename)
