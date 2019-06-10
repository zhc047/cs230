import tensorflow as tf
import stylegan.dnnlib as dnnlib
import sys
sys.modules["dnnlib"] = dnnlib
from dnnlib.tflib.tfutil import convert_images_to_uint8
import stylegan.dnnlib.tflib as tflib
import stylegan.config as config
from stylegan.training import misc
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
experiment_dir = './experiments/' + sys.argv[1]
os.mkdir(experiment_dir)
def write_summary(value, tag, global_step, summary_writer):
    summary = tf.Summary()
    summary.value.add(tag=tag, simple_value=value)
    summary_writer.add_summary(summary, global_step)

summary_epoch = 500
num_styles = 18
dlatent_size = 512
lamda = 0
num_epochs = 2000
learning_rate = 1e-2
intp_lamda = 0.7
syn_loss_lamda = 0
epsilon = 1e-4

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

Ws = []
print(W.shape)
for i in range(num_styles):
  print(W[:, i, :].shape)
  tiled_temp = tf.tile(tf.expand_dims(W[:, i, :], 1), [1, 18 // num_styles, 1])
  print(tiled_temp.shape)
  Ws.append(tiled_temp)

W_tiled = tf.concat(Ws, 1)
print(W_tiled.shape)

generated_img = Gs.components.synthesis.get_output_for(W_tiled, randomize_noise=True, structure='linear')
tf.summary.image('images/generated', convert_images_to_uint8(generated_img, nchw_to_nhwc=True))
generated_img = tf.transpose(generated_img, perm=[0, 2, 3, 1])
tf.summary.histogram('float_imgs/generated', generated_img)

# W_synthesis_coarse = intp_lamda * W_tiled[0, :, :int(dlatent_size / 2)] + (1 - intp_lamda) * W_tiled[1, :, :int(dlatent_size / 2)]
# W_synthesis_fine = (1 - intp_lamda) * W_tiled[0, :, int(dlatent_size / 2):] + intp_lamda * W_tiled[1, :, int(dlatent_size / 2):]
# W_synthesis = tf.concat([W_synthesis_coarse, W_synthesis_fine], 1)

# W_synthesis_coarse = intp_lamda * W[0][:4] + (1 - intp_lamda) * W[1][:4]
# W_synthesis_fine = (1 - intp_lamda) * W[0][4:] + intp_lamda * W[1][4:]
# W_synthesis = tf.concat([W_synthesis_coarse, W_synthesis_fine], 0)

W_synthesis = intp_lamda * W_tiled[0] + (1 - intp_lamda) * W_tiled[1]


# W_synthesis_coarse_epsilon = (intp_lamda + epsilon )* W_tiled[0, :, :256] + (1 - intp_lamda - epsilon) * W_tiled[1, :, :256]
# W_synthesis_fine_epsilon = (1 - intp_lamda - epsilon) * W_tiled[0, :, 256:] + (intp_lamda + epsilon) * W_tiled[1, :, 256:]
# W_synthesis_epsilon = tf.concat([W_synthesis_coarse_epsilon, W_synthesis_fine_epsilon], 1)

# W_synthesis_coarse_epsilon = (intp_lamda + epsilon )* W_tiled[0][:4] + (1 - intp_lamda - epsilon) * W_tiled[1][:4]
# W_synthesis_fine_epsilon = (1 - intp_lamda - epsilon) * W_tiled[0][4:] + (intp_lamda + epsilon) * W_tiled[1][4:]
# W_synthesis_epsilon = tf.concat([W_synthesis_coarse_epsilon, W_synthesis_fine_epsilon], 0)

W_synthesis_epsilon = (intp_lamda + epsilon) * W_tiled[0] + (1 - intp_lamda - epsilon) * W_tiled[1]

W_synthesis = tf.expand_dims(W_synthesis, 0)
W_synthesis_epsilon = tf.expand_dims(W_synthesis_epsilon, 0)
img_synthesis = Gs.components.synthesis.get_output_for(W_synthesis)
img_synthesis_epsilon = Gs.components.synthesis.get_output_for(W_synthesis_epsilon)
imgs_synthesis = tf.concat([img_synthesis, img_synthesis_epsilon], 0)
print(imgs_synthesis.shape)

# Crop only the face region.
c = int(imgs_synthesis.shape[2] // 8)
imgs_synthesis = imgs_synthesis[:, :, c*3 : c*7, c*2 : c*6]

# Downsample image to 256x256 if it's larger than that. VGG was built for 224x224 images.
if imgs_synthesis.shape[2] > 256:
    factor = imgs_synthesis.shape[2] // 256
    imgs_synthesis = tf.reshape(imgs_synthesis, [-1, imgs_synthesis.shape[1], imgs_synthesis.shape[2] // factor, factor, imgs_synthesis.shape[3] // factor, factor])
    imgs_synthesis = tf.reduce_mean(imgs_synthesis, axis=[3,5])

# Scale dynamic range from [-1,1] to [0,255] for VGG.
imgs_synthesis = (imgs_synthesis + 1) * (255 / 2)

# Evaluate perceptual distance.
img_e0, img_e1 = imgs_synthesis[0::2], imgs_synthesis[1::2]
# tf.summary.image('cropped/origin', tf.transpose(img_e0, [0, 2, 3, 1]))
# tf.summary.image('cropped/epsilon', tf.transpose(img_e1, [0, 2, 3, 1]))
distance_measure = misc.load_pkl('https://drive.google.com/uc?id=1N2-m9qszOeVC9Tq77WxsLnuWwOedQiD2') # vgg16_zhang_perceptual.pkl
ppl = distance_measure.get_output_for(img_e0, img_e1) * (1 / epsilon **2)

fake_scores_out = fp32(D.get_output_for(img_synthesis, None))
img_synthesis = convert_images_to_uint8(img_synthesis, nchw_to_nhwc=True)
tf.summary.image('images/synthesis', img_synthesis)

real_img1 = load_img('ffhq.png')
real_img2 = load_img('celeba.png')
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
loss += syn_loss_lamda * (1 - fake_scores_out)
loss = tf.squeeze(loss)

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
            input_feed.append(ppl)
            input_feed.append(merge_summaries)
        outputs = sess.run(input_feed)
        loss_val = outputs[1]

        print("epoch " + str(i) + ": " + str(loss_val) + "\n")
        write_summary(loss_val, 'loss', i, summary_writer)
        if i % summary_epoch == 0:
            summary_writer.add_summary(outputs[3], i+1)
            ppl_val = outputs[2]
            write_summary(ppl_val, 'ppl', i, summary_writer)

    optimal_img = Gs.components.synthesis.get_output_for(W_tiled)
    tf.summary.histogram('output_img/float', optimal_img)
    optimal_img = convert_images_to_uint8(optimal_img, nchw_to_nhwc=True)
    tf.summary.histogram('output_img/int', optimal_img)
    optimal_img, summaries = sess.run([optimal_img, merge_summaries])
    summary_writer.add_summary(summaries, i)

    for i, img in enumerate(optimal_img):
        png_filename = os.path.join(experiment_dir, 'optimal{}.png'.format(i))
        PIL.Image.fromarray(optimal_img[i], 'RGB').save(png_filename)
    # todo
    W_synthesis = intp_lamda * W_tiled[0] + (1 - intp_lamda) * W_tiled[1]
    W_synthesis = tf.expand_dims(W_synthesis, 0)
    img_synthesis = Gs.components.synthesis.get_output_for(W_synthesis)
    img_synthesis = convert_images_to_uint8(img_synthesis, nchw_to_nhwc=True)
    img_synthesis = sess.run([img_synthesis])

    png_filename = os.path.join(experiment_dir, 'synthesis.png')
    PIL.Image.fromarray(img_synthesis[0][0], 'RGB').save(png_filename)
