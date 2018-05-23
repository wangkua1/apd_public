import tensorflow as tf
import os


def initial_tensorboard(exp_name):
    sess = tf.InteractiveSession()
    if not os.path.exists('logs'):
        os.mkdir('logs')
    log_folder = os.path.join('logs', exp_name)
    # remove existing log folder for the same model.
    if os.path.exists(log_folder):
        import shutil
        shutil.rmtree(log_folder, ignore_errors=True)


    train_writer = tf.summary.FileWriter(os.path.join(log_folder, 'train'), sess.graph)
    point_writer = tf.summary.FileWriter(os.path.join(log_folder, 'validation_point'), sess.graph)
    posterior_writer = tf.summary.FileWriter(os.path.join(log_folder, 'validation_posterior'), sess.graph)
    mcdropout_writer = tf.summary.FileWriter(os.path.join(log_folder, 'validation_mcdropout'), sess.graph)

    return sess, train_writer, point_writer, posterior_writer, mcdropout_writer