#!/usr/bin/env python

import argparse
import data_utils
import gan
import gan_utils
import glob
import os
import time
import tqdm

from datetime import datetime

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

print('Tensorflow version:', tf.__version__)

os.environ["OMP_NUM_THREADS"]="4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"   
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.keras.backend.set_floatx('float32')

start_time = time.time()

def train(args):
    test = args.test
    dname = args.dname
    time_steps = args.time_steps
    batch_size = args.batch_size
    path = args.path
    print(path)
    seed = args.seed
    save_freq = args.save_freq
    
    # filter size for (de)convolutional layers
    g_state_size = args.g_state_size
    d_state_size = args.d_state_size
    g_filter_size = args.g_filter_size
    d_filter_size = args.d_filter_size
    reg_penalty = args.reg_penalty
    nlstm = args.n_lstm
    x_width = 64
    x_height = 64
    channels = args.n_channels
    epochs = 1
    buffer = 200
    bn = args.batch_norm
    projection = args.projector

    # path to data
    filenames = glob.glob(path)
    data_processor = data_utils.DataProcessor(
        filenames, time_steps, channels)
    batched_x = data_processor.provide_video_data(
        buffer, batch_size * 2, x_height, x_width)

    # adjust channel parameter as we want to drop the
    # alpha channel for animated Sprites
    if dname == 'animation':
        channels = channels - 1

    dataset = dname + '-cot'
    # Number of RNN layers stacked together
    n_layers = 1
    gen_lr = args.lr
    disc_lr = args.lr
    tf.random.set_seed(seed)
    np.random.seed(seed)
    # decaying learning rate scheme
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=gen_lr, decay_steps=500,
        decay_rate=0.98, staircase=True)
    # Add gradient clipping before updates
    gen_optimiser = tf.keras.optimizers.Adam(lr_schedule,
                                             beta_1=0.5,
                                             beta_2=0.9)
    dischm_optimiser = tf.keras.optimizers.Adam(lr_schedule,
                                                beta_1=0.5,
                                                beta_2=0.9)

    it_counts = 0
    disc_iters = 1
    sinkhorn_eps = args.sinkhorn_eps
    sinkhorn_l = args.sinkhorn_l
    time_steps = args.time_steps
    scaling_coef = 1.0

    # Create instances of generator, discriminator_h and
    # discriminator_m CONV VERSION
    z_width = args.z_dims_t
    z_height = args.z_dims_t

    # Define a standard multivariate normal for (z1, z2, ..., zT) --> (y1, y2, ..., yT)
    dist_z = tfp.distributions.Normal(0.0, 1.0)
    # parameters for fixed latents
    y_dims = args.y_dims
    dist_y = tfp.distributions.Normal(0.0, 1.0)

    generator = gan.VideoDCG(
        batch_size, time_steps, x_width, x_height, z_width, z_height, g_state_size,
        filter_size=g_filter_size, bn=bn, nlstm=nlstm, nchannel=channels)

    discriminator_h = gan.VideoDCD(
        batch_size, time_steps, x_width, x_height, z_width, z_height, d_state_size,
        filter_size=d_filter_size, bn=bn, nchannel=channels)
    discriminator_m = gan.VideoDCD(
        batch_size, time_steps, x_width, x_height, z_width, z_height, d_state_size,
        filter_size=d_filter_size, bn=bn, nchannel=channels)

    # data_utils.check_model_summary(batch_size, z_dims, generator)
    # data_utils.check_model_summary(batch_size, seq_len, discriminator_h)

    saved_file = "{}_{}{}-{}:{}:{}.{}".format(dataset,
                                              datetime.now().strftime("%h"),
                                              datetime.now().strftime("%d"),
                                              datetime.now().strftime("%H"),
                                              datetime.now().strftime("%M"),
                                              datetime.now().strftime("%S"),
                                              datetime.now().strftime("%f"))

    model_fn = "%s_Dz%d_Dy%d_bs%d_gss%d_gfs%d_dss%d_dfs%d_eps%d_l%d_p%d_lr%d_nl%d_s%02d" % \
               (dname, args.z_dims_t, args.y_dims, batch_size,
                g_state_size, g_filter_size,
                d_state_size, d_filter_size,
                np.round(np.log10(sinkhorn_eps)), sinkhorn_l,
                np.round(np.log10(reg_penalty)),
                np.round(np.log10(gen_lr)), nlstm, seed)

    log_dir = "./trained/{}/log".format(saved_file)

    # Create directories for storing images later.
    if not os.path.exists("trained/{}/data".format(saved_file)):
        os.makedirs("trained/{}/data".format(saved_file))
    if not os.path.exists("trained/{}/images".format(saved_file)):
        os.makedirs("trained/{}/images".format(saved_file))

    # GAN train notes
    with open("./trained/{}/train_notes.txt".format(saved_file), 'w') as f:
        # Include any experiment notes here:
        f.write("Experiment notes: .... \n\n")
        f.write("MODEL_DATA: {}\nSEQ_LEN: {}\n".format(
            dataset,
            time_steps, ))
        f.write("STATE_SIZE: {}\nNUM_LAYERS: {}\nLAMBDA: {}\n".format(
            g_state_size,
            n_layers,
            reg_penalty))
        f.write("BATCH_SIZE: {}\nCRITIC_ITERS: {}\nGenerator LR: {}\nDiscriminator LR:{}\n".format(
            batch_size,
            disc_iters,
            gen_lr,
            disc_lr))
        f.write("SINKHORN EPS: {}\nSINKHORN L: {}\n\n".format(
            sinkhorn_eps,
            sinkhorn_l))

    train_writer = tf.summary.create_file_writer(logdir=log_dir)

    @tf.function
    def disc_training_step(real_x, real_x_p):
        hidden_z = dist_z.sample([batch_size, time_steps, z_width * z_height])
        hidden_z_p = dist_z.sample([batch_size, time_steps, z_width * z_height])

        hidden_y = dist_y.sample([batch_size, y_dims])
        hidden_y_p = dist_y.sample([batch_size, y_dims])

        with tf.GradientTape() as disc_tape:
            fake_data = generator.call(hidden_z, hidden_y)
            fake_data_p = generator.call(hidden_z_p, hidden_y_p)

            h_fake = discriminator_h.call(fake_data)

            m_real = discriminator_m.call(real_x)
            m_fake = discriminator_m.call(fake_data)

            h_real_p = discriminator_h.call(real_x_p)
            h_fake_p = discriminator_h.call(fake_data_p)

            m_real_p = discriminator_m.call(real_data_p)

            loss1 = gan_utils.compute_mixed_sinkhorn_loss(
                real_data, fake_data, m_real, m_fake, h_fake, scaling_coef,
                sinkhorn_eps, sinkhorn_l, real_data_p, fake_data_p, m_real_p,
                h_real_p, h_fake_p)
            pm1 = gan_utils.scale_invariante_martingale_regularization(
                m_real, reg_penalty, scaling_coef)
            disc_loss = - loss1 + pm1

        # update discriminator parameters
        disch_grads, discm_grads = disc_tape.gradient(
            disc_loss, [discriminator_h.trainable_variables, discriminator_m.trainable_variables])
        dischm_optimiser.apply_gradients(zip(disch_grads, discriminator_h.trainable_variables))
        dischm_optimiser.apply_gradients(zip(discm_grads, discriminator_m.trainable_variables))

    @tf.function
    def gen_training_step(real_x, real_x_p):
        hidden_z = dist_z.sample([batch_size, time_steps, z_width * z_height])
        hidden_z_p = dist_z.sample([batch_size, time_steps, z_width * z_height])

        hidden_y = dist_y.sample([batch_size, y_dims])
        hidden_y_p = dist_y.sample([batch_size, y_dims])

        with tf.GradientTape() as gen_tape:
            fake_data = generator.call(hidden_z, hidden_y)
            fake_data_p = generator.call(hidden_z_p, hidden_y_p)

            h_fake = discriminator_h.call(fake_data)

            m_real = discriminator_m.call(real_x)
            m_fake = discriminator_m.call(fake_data)

            h_real_p = discriminator_h.call(real_x_p)
            h_fake_p = discriminator_h.call(fake_data_p)

            m_real_p = discriminator_m.call(real_data_p)

            loss2 = gan_utils.compute_mixed_sinkhorn_loss(
                real_data, fake_data, m_real, m_fake, h_fake, scaling_coef,
                sinkhorn_eps, sinkhorn_l, real_data_p, fake_data_p, m_real_p,
                h_real_p, h_fake_p)

            gen_loss = loss2
        # update generator parameters
        generator_grads = gen_tape.gradient(gen_loss, generator.trainable_variables)
        gen_optimiser.apply_gradients(zip(generator_grads, generator.trainable_variables))
        return loss2

    with tqdm.trange(epochs, ncols=100, unit="epoch") as ep:
        for _ in ep:
            it = tqdm.tqdm(ncols=100)
            for x in batched_x:
                if x.shape[0] != batch_size*2:
                    continue
                it_counts += 1
                # split the batches for x and x'
                real_data = x[0:batch_size, ]
                real_data_p = x[batch_size:, ]

                real_data = tf.reshape(real_data, [batch_size, x_height, x_width * time_steps, -1])
                real_data_p = tf.reshape(real_data_p, [batch_size, x_height, x_width * time_steps, -1])
                # throw away alpha channel
                real_data = real_data[..., :channels]
                real_data_p = real_data_p[..., :channels]
                disc_training_step(real_data, real_data_p)
                loss = gen_training_step(real_data, real_data_p)
                it.set_postfix(loss=float(loss))
                it.update(1)

                with train_writer.as_default():
                    tf.summary.scalar('Total Mixed Sinkhorn Loss', loss, step=it_counts)
                    train_writer.flush()

                if not np.isfinite(loss.numpy()):
                    print('%s Loss exploded!' % model_fn)
                    # Open the existing file with mode a - append
                    with open("./trained/{}/train_notes.txt".format(saved_file), 'a') as f:
                        # Include any experiment notes here:
                        f.write("\n Training failed! ")
                    break
                else:
                    if it_counts % save_freq == 0 or it_counts == 1:
                        z = dist_z.sample([batch_size, time_steps, z_width * z_height])
                        y = dist_y.sample([batch_size, y_dims])
                        samples = generator.call(z, y, training=False)
                        # plot first 10 samples within one image
                        img = tf.concat(list(samples[:10]), axis=0)[None]
                        with train_writer.as_default():
                            tf.summary.image("Training data", img, step=it_counts)
                        # save model to file
                        generator.save_weights("./trained/{}/{}/".format(test, model_fn))
                        discriminator_h.save_weights("./trained/{}/{}_h/".format(test, model_fn))
                        discriminator_m.save_weights("./trained/{}/{}_m/".format(test, model_fn))
                continue
    print("--- The entire training takes %s minutes ---" % ((time.time() - start_time) / 60.0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='cot')
    
    parser.add_argument('-d', '--dname', type=str, default='animation',
                        choices=['animation', 'human_action'])
    parser.add_argument('-t', '--test',  type=str, default='cot', choices=['cot'])
    parser.add_argument('-s', '--seed', type=int, default=1)
    parser.add_argument('-gss', '--g_state_size', type=int, default=32)
    parser.add_argument('-gfs', '--g_filter_size', type=int, default=32)
    parser.add_argument('-dss', '--d_state_size', type=int, default=32)
    parser.add_argument('-dfs', '--d_filter_size', type=int, default=32)
    
    # animation data has T=13 and human action data has T=16
    parser.add_argument('-ts', '--time_steps', type=int, default=13)
    parser.add_argument('-sinke', '--sinkhorn_eps', type=float, default=0.8)
    parser.add_argument('-reg_p', '--reg_penalty', type=float, default=0.01)
    parser.add_argument('-sinkl', '--sinkhorn_l', type=int, default=100)
    parser.add_argument('-Dx', '--Dx', type=int, default=1)
    parser.add_argument('-Dz', '--z_dims_t', type=int, default=5)
    parser.add_argument('-Dy', '--y_dims', type=int, default=20)
    parser.add_argument('-g', '--gen', type=str, default="fc",
                        choices=["lstm", "fc"])
    parser.add_argument('-bs', '--batch_size', type=int, default=4)
    parser.add_argument('-p', '--path', type=str,
                        default='/home/tianlin_xu/dataset/animation/*_data.tfrecord')
    parser.add_argument('-save', '--save_freq', type=int, default=500)
    parser.add_argument('-lr', '--lr', type=float, default=1e-3)
    parser.add_argument('-bn', '--batch_norm', type=bool, default=True)
    parser.add_argument('-nlstm', '--n_lstm', type=int, default=1)
    
    # animation original data has 4 channels and human action data has 3
    parser.add_argument('-nch', '--n_channels', type=int, default=4)
    parser.add_argument('-rt', '--read_tfrecord', type=bool, default=True)
    parser.add_argument('-lp', '--projector', type=bool, default=False)

    args = parser.parse_args()
    
    train(args)
