import matplotlib as mpl

mpl.use("Agg")
import os
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import data_utils
import time
import gan_utils
import gan
import math
from datetime import datetime
import argparse, tqdm
from functools import partial

start_time = time.time()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.keras.backend.set_floatx('float32')

parser = argparse.ArgumentParser(description='cot')
parser.add_argument('-d', '--dname', type=str, default='SineImage', choices=['SineImage', 'AROne', 'eeg'])
parser.add_argument('-t', '--test', type=str, default='cot', choices=['cot'])
parser.add_argument('-s', '--seed', type=int, default=42)
parser.add_argument('-gss', '--g_state_size', type=int, default=32)
parser.add_argument('-dss', '--d_state_size', type=int, default=32)
parser.add_argument('-gfs', '--g_filter_size', type=int, default=32)
parser.add_argument('-dfs', '--d_filter_size', type=int, default=32)
parser.add_argument('-r', '--reg_penalty', type=float, default=10.0)
parser.add_argument('-ts', '--time_steps', type=int, default=48)
parser.add_argument('-sinke', '--sinkhorn_eps', type=float, default=100)
parser.add_argument('-sinkl', '--sinkhorn_l', type=int, default=100)
parser.add_argument('-Dy', '--Dy', type=int, default=10)
parser.add_argument('-Dz', '--z_dims_t', type=int, default=10)
parser.add_argument('-g', '--gen', type=str, default="lstm", choices=["lstm", "fc"])
parser.add_argument('-bs', '--batch_size', type=int, default=32)
parser.add_argument('-nlstm', '--nlstm', type=int, default=1, help="number of lstms in discriminator")
parser.add_argument('-lr', '--lr', type=float, default=1e-3)
parser.add_argument('-bn', '--bn', type=int, default=1, help="batch norm")

args = parser.parse_args()


def main():
    #
    # hyper-parameter settings
    #
    dname = args.dname
    test = args.test
    time_steps = args.time_steps
    batch_size = args.batch_size
    bn = bool(args.bn)
    if "SLURM_ARRAY_TASK_ID" in os.environ:
        seed = int(os.environ["SLURM_ARRAY_TASK_ID"])
    else:
        seed = args.seed

    n_iters = 100000
    Dx = 10
    g_output_activation = 'linear'
    time_steps = 100

    if dname == 'AROne':
        data_dist = data_utils.AROne(Dx, time_steps, np.linspace(0.1, 0.9, Dx), 0.5)
    elif dname == 'eeg':
        data_dist = data_utils.EEGData(Dx, time_steps, batch_size, n_iters, seed=seed)
    elif dname == 'SineImage':
        data_dist = data_utils.SineImage(length=time_steps, Dx=Dx, rand_std=0.1)
    else:
        ValueError('Data does not exist.')

    dataset = dname
    # Number of RNN layers stacked together
    n_layers = 1
    reg_penalty = args.reg_penalty
    gen_lr = args.lr
    disc_lr = args.lr
    tf.random.set_seed(seed)
    np.random.seed(seed)
    # Add gradient clipping before updates
    gen_optimiser = tf.keras.optimizers.Adam(gen_lr)
    dischm_optimiser = tf.keras.optimizers.Adam(disc_lr)

    it_counts = 0
    disc_iters = 1
    sinkhorn_eps = args.sinkhorn_eps
    sinkhorn_l = args.sinkhorn_l
    nlstm = args.nlstm
    scaling_coef = 1.0

    # Define a standard multivariate normal for (z1, z2, ..., zT) --> (y1, y2, ..., yT)
    z_dims_t = args.z_dims_t
    y_dims = args.Dy
    dist_z = tfp.distributions.Uniform(-1, 1)
    dist_y = tfp.distributions.Uniform(-1, 1)

    # Create instances of generator, discriminator_h and discriminator_m CONV VERSION
    g_state_size = args.g_state_size
    d_state_size = args.d_state_size
    g_filter_size = args.g_filter_size
    d_filter_size = args.d_filter_size
    disc_kernel_width = 5

    if args.gen == "fc":
        generator = gan.SimpleGenerator(batch_size, time_steps, Dx, g_filter_size, z_dims_t,
                                        output_activation=g_output_activation)
    elif args.gen == "lstm":
        generator = gan.ToyGenerator(batch_size, time_steps, z_dims_t, Dx, g_state_size, g_filter_size,
                                     output_activation=g_output_activation, nlstm=nlstm, nlayer=2, Dy=y_dims, bn=bn)

    discriminator_h = gan.ToyDiscriminator(batch_size, time_steps, z_dims_t, Dx, d_state_size, d_filter_size,
                                           kernel_size=disc_kernel_width, nlayer=2, nlstm=0, bn=bn)
    discriminator_m = gan.ToyDiscriminator(batch_size, time_steps, z_dims_t, Dx, d_state_size, d_filter_size,
                                           kernel_size=disc_kernel_width, nlayer=2, nlstm=0, bn=bn)

    # data_utils.check_model_summary(batch_size, z_dims, generator)
    # data_utils.check_model_summary(batch_size, seq_len, discriminator_h)

    lsinke = int(np.round(np.log10(sinkhorn_eps)))
    lreg = int(np.round(np.log10(reg_penalty)))
    saved_file = f"{dname[:3]}_{test[0]}_e{lsinke:d}r{lreg:d}s{seed:d}" + \
                 "{}_{}{}-{}:{}:{}.{}".format(dataset, datetime.now().strftime("%h"),datetime.now().strftime("%d"),
                                              datetime.now().strftime("%H"), datetime.now().strftime("%M"),
                                              datetime.now().strftime("%S"), datetime.now().strftime("%f"))

    model_fn = "%s_Dz%d_Dy%d_Dx%d_bs%d_gss%d_gfs%d_dss%d_dfs%d_ts%d_r%d_eps%d_l%d_lr%d_nl%d_s%02d" % (
        dname, z_dims_t, y_dims, Dx, batch_size, g_state_size, g_filter_size, d_state_size, d_filter_size,
        time_steps, np.round(np.log10(reg_penalty)), np.round(np.log10(sinkhorn_eps)),
        sinkhorn_l, np.round(np.log10(args.lr)), nlstm, seed)

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

    with train_writer.as_default():
        tf.summary.text('model_fn', model_fn, step=1)

    @tf.function
    def disc_training_step(real_data, real_data_p):
        hidden_z = dist_z.sample([batch_size, time_steps, z_dims_t])
        hidden_z_p = dist_z.sample([batch_size, time_steps, z_dims_t])
        hidden_y = dist_y.sample([batch_size, y_dims])
        hidden_y_p = dist_y.sample([batch_size, y_dims])

        with tf.GradientTape(persistent=True) as disc_tape:
            fake_data = generator.call(hidden_z, hidden_y)
            fake_data_p = generator.call(hidden_z_p, hidden_y_p)

            h_fake = discriminator_h.call(fake_data)

            m_real = discriminator_m.call(real_data)
            m_fake = discriminator_m.call(fake_data)

            h_real_p = discriminator_h.call(real_data_p)
            h_fake_p = discriminator_h.call(fake_data_p)

            m_real_p = discriminator_m.call(real_data_p)

            loss1 = gan_utils.compute_mixed_sinkhorn_loss(real_data, fake_data, m_real, m_fake, h_fake, scaling_coef,
                                                          sinkhorn_eps, sinkhorn_l, real_data_p, fake_data_p, m_real_p,
                                                          h_real_p, h_fake_p)
            pm1 = gan_utils.scale_invariante_martingale_regularization(m_real, reg_penalty, scaling_coef)
            disc_loss = - loss1 + pm1
        # update discriminator parameters
        disch_grads, discm_grads = disc_tape.gradient(disc_loss, [discriminator_h.trainable_variables,
                                                                  discriminator_m.trainable_variables])
        dischm_optimiser.apply_gradients(zip(disch_grads, discriminator_h.trainable_variables))
        dischm_optimiser.apply_gradients(zip(discm_grads, discriminator_m.trainable_variables))

    @tf.function
    def gen_training_step(real_data, real_data_p):
        hidden_z = dist_z.sample([batch_size, time_steps, z_dims_t])
        hidden_z_p = dist_z.sample([batch_size, time_steps, z_dims_t])
        hidden_y = dist_y.sample([batch_size, y_dims])
        hidden_y_p = dist_y.sample([batch_size, y_dims])

        with tf.GradientTape() as gen_tape:
            fake_data = generator.call(hidden_z, hidden_y)
            fake_data_p = generator.call(hidden_z_p, hidden_y_p)

            h_fake = discriminator_h.call(fake_data)

            m_real = discriminator_m.call(real_data)
            m_fake = discriminator_m.call(fake_data)

            h_real_p = discriminator_h.call(real_data_p)
            h_fake_p = discriminator_h.call(fake_data_p)

            m_real_p = discriminator_m.call(real_data_p)

            loss2 = gan_utils.compute_mixed_sinkhorn_loss(real_data, fake_data, m_real, m_fake, h_fake, scaling_coef,
                                                          sinkhorn_eps, sinkhorn_l, real_data_p, fake_data_p, m_real_p,
                                                          h_real_p, h_fake_p)
            gen_loss = loss2
        # update generator parameters
        generator_grads = gen_tape.gradient(gen_loss, generator.trainable_variables)
        gen_optimiser.apply_gradients(zip(generator_grads, generator.trainable_variables))
        return loss2

    with tqdm.trange(n_iters, ncols=150) as it:
        for _ in it:
            it_counts += 1
            # generate a batch of REAL data
            real_data = data_dist.batch(batch_size)
            real_data_p = data_dist.batch(batch_size)
            real_data = tf.cast(real_data, tf.float32)
            real_data_p = tf.cast(real_data_p, tf.float32)

            disc_training_step(real_data, real_data_p)
            loss = gen_training_step(real_data, real_data_p)
            it.set_postfix(loss=float(loss))

            with train_writer.as_default():
                tf.summary.scalar('Sinkhorn loss', loss, step=it_counts)
                train_writer.flush()

            if not np.isfinite(loss.numpy()):
                print('%s Loss exploded!' % model_fn)
                # Open the existing file with mode a - append
                with open("./trained/{}/train_notes.txt".format(saved_file), 'a') as f:
                    # Include any experiment notes here:
                    f.write("\n Training failed! ")
                break
            else:
                if it_counts % 100 == 0 or it_counts <= 10:
                    # print("---  Plot samples produced by generator after %d iterations ---" % it_counts)
                    z = dist_z.sample([batch_size, time_steps, z_dims_t])
                    y = dist_y.sample([batch_size, y_dims])
                    samples = generator.call(z, y, training=False)
                    # save plot to file
                    if samples.shape[-1] == 1:
                        data_utils.plot_batch(np.asarray(samples[..., 0]), it_counts, saved_file)

                    img = tf.transpose(tf.concat(list(samples[:5]), axis=1))[None, :, :, None]
                    with train_writer.as_default():
                        tf.summary.image("Training data", img, step=it_counts)
                    # save model to file
                    generator.save_weights("./trained/{}/{}/".format(test, model_fn))
                    discriminator_h.save_weights("./trained/{}/{}_h/".format(test, model_fn))
                    discriminator_m.save_weights("./trained/{}/{}_m/".format(test, model_fn))
            continue

    print("--- The entire training takes %s minutes ---" % ((time.time() - start_time) / 60.0))


if __name__ == '__main__':
    main()