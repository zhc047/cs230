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
# from tensorflow.contrib.slim.preprocessing.vgg_preprocessing import preprocess_image
slim = tf.contrib.slim

num_styles = 18
dlatent_size = 512
lamda = 1
num_epochs = 100

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
W = tf.get_variable('W', initializer=W_np)

generated_img = Gs.components.synthesis.get_output_for(W)
# generated_img = tf.transpose(generated_img, perm=[0, 3, 2, 1])
generated_img = big_generated_img = convert_images_to_uint8(generated_img, nchw_to_nhwc=True)
generated_img = tf.image.resize_images(generated_img, size=[224,224])
generated_img = preprocess_input(generated_img)
# generated_img = slim.preprocessing.vgg_preprocessing.preprocess_image(generated_img)

real_img = load_img('example.png')
real_img = tf.constant(real_img)
real_img = big_real_img = tf.expand_dims(real_img, 0)
# big_real_img = tf.cast(big_real_img, tf.float32)
real_img = tf.image.resize_images(real_img, size=[224, 224])
real_img = preprocess_input(real_img)

# _, end_points_real = vgg.vgg_16(real_img)
# _, end_points_generated = vgg.vgg_16(generated_img)
#
# layers = ['vgg_16/conv1/conv1_1', 'vgg_16/conv1/conv1_2', 'vgg_16/conv3/conv3_2', 'vgg_16/conv4/conv4_2']

loss = pix_loss = lamda * tf.norm(big_real_img - big_generated_img)
# for layer in layers:
#     activation_generated = end_points_generated[layer]
#     activation_real = end_points_real[layer]
#     loss += tf.norm(activation_generated - activation_real)

optim = tf.train.AdamOptimizer(learning_rate=0.01)
train_op = optim.minimize(loss, var_list=[W])

with tf.Session() as sess:
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    # restore = slim.assign_from_checkpoint_fn(
    #            'vgg_16.ckpt',
    #            slim.get_model_variables("vgg_16")
    #           )
    sess.run(init_op)
    # restore(sess)
    for i in range(num_epochs):
        _, loss_val, pix_loss_val = sess.run([train_op, loss, pix_loss])
        print("epoch " + str(i) + ": " + str(loss_val) + "\n")
        # print(pix_loss_val)

    optimal_img = Gs.components.synthesis.get_output_for(W)
    optimal_img = convert_images_to_uint8(optimal_img, nchw_to_nhwc=True)
    optimal_img = sess.run(optimal_img)
    print(np.max(optimal_img), np.min(optimal_img))
    # optimal_img = np.transpose(optimal_img, [0, 3, 2, 1])

    os.makedirs(config.result_dir, exist_ok=True)
    png_filename = os.path.join(config.result_dir, 'optimal.png')
    PIL.Image.fromarray(optimal_img[0], 'RGB').save(png_filename)
