import os
import time
import random
import logging
import argparse
import keras
import numpy as np
import tensorflow as tf

from cleverhans import attacks
from cleverhans import utils, utils_tf, utils_keras

from resnet import resnet

import warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ["CUDA_VISIBLE_DEVICES"] = '2'


class Dataset:
    def __init__(self, path):
        self.train_data = np.load(os.path.join(path, "train_data.npy"))
        self.test_data = np.load(os.path.join(path, "test_data_100.npy"))
        self.train_label = np.load(os.path.join(path, "train_label.npy"))
        self.test_label = np.load(os.path.join(path, "test_label_100.npy"))


def get_random_targets(label):
    targets = label.copy()
    nb_samples = label.shape[0]
    nb_classes = label.shape[1]

    for i in range(nb_samples):
        targets[i, :] = np.roll(targets[i, :], random.randint(1, nb_classes - 1))

    return targets


def adversarial_attack(data, args):
    # Set attack parameters
    eps = float(args.eps)
    order = args.lorder
    batch_size = 100

    # Set evaluation parameters
    eval_params = {'batch_size': batch_size}

    # Object used to keep track of (and return) key accuracies
    report = utils.AccuracyReport()

    # Set TF random seed to improve reproducibility
    tf.set_random_seed(1234)

    # Set logging level to see debug information
    utils.set_log_level(logging.DEBUG)

    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    sess = tf.Session(config=tf_config)
    keras.backend.set_session(sess)

    # Get CIFAR10 data
    x_train, y_train = data.train_data, data.train_label
    x_test, y_test = data.test_data, data.test_label

    img_rows, img_cols, nchannels = x_train.shape[1:4]
    nb_classes = y_train.shape[1]

    # Define input TF placeholders
    x = tf.placeholder(tf.float32, shape=(None, img_rows, img_cols, nchannels))
    y = tf.placeholder(tf.float32, shape=(None, nb_classes))

    def evaluate(preds, x_set, y_set, report_key, is_adv=None):
        acc = utils_tf.model_eval(sess, x, y, preds, x_set, y_set, args=eval_params)
        setattr(report, report_key, acc)
        if is_adv is None:
            report_text = None
        elif is_adv:
            report_text = 'adversarial'
        else:
            report_text = 'legitimate'
        if report_text:
            print('Test accuracy on %s examples: %0.4f' % (report_text, acc))

    # Load baseline model
    model = resnet(input_shape=x_train.shape[1:], depth=29, num_classes=10).build()
    model.load_weights(args.model)

    wrapper = utils_keras.KerasModelWrapper(model)
    preds = model(x)
    evaluate(preds, x_test, y_test, 'clean_train_clean_eval', False)

    if args.attack == 'fgsm':
        # Fast Gradient Sign Method (FGSM) attack
        fgsm_params = {'eps': eps, 'ord': order, 'clip_min': 0., 'clip_max': 1.}
        fgsm = attacks.FastGradientMethod(wrapper, sess=sess)
        adv_x = fgsm.generate(x, **fgsm_params)
        preds_adv = model(adv_x)
    elif args.attack == 'ifgsm':
        # Fast Gradient Sign Method (FGSM) attack
        ifgsm_params = {'eps': eps, 'eps_iter': eps / 10, 'ord': order, 'clip_min': 0., 'clip_max': 1.}
        ifgsm = attacks.BasicIterativeMethod(wrapper, sess=sess)
        adv_x = ifgsm.generate(x, **ifgsm_params)
        preds_adv = model(adv_x)
    elif args.attack == 'mifgsm':
        mifgsm_params = {'eps': eps, 'eps_iter': eps / 10, 'ord': order, 'clip_min': 0., 'clip_max': 1.}
        mifgsm = attacks.MomentumIterativeMethod(wrapper, sess=sess)
        adv_x = mifgsm.generate(x, **mifgsm_params)
        preds_adv = model(adv_x)
    elif args.attack == 'jsma':
        jsma_params = {'theta': 1., 'gamma': 1., 'ord': order, 'clip_min': 0., 'clip_max': 1.}
        jsma = attacks.SaliencyMapMethod(wrapper, sess=sess)
        adv_x = jsma.generate(x, **jsma_params)
        preds_adv = model(adv_x)
    elif args.attack == 'lbfgs':
        # y_target = tf.placeholder(tf.float32, shape=(None, nb_classes))
        lbfgs_params = {'y_target': tf.convert_to_tensor(get_random_targets(data.test_label)), 'batch_size': batch_size,
                        'binary_search_steps': 4, 'max_iterations': 1000,
                        'clip_min': 0., 'clip_max': 1.}
        lbfgs = attacks.LBFGS(wrapper, sess=sess)
        adv_x = lbfgs.generate(x, **lbfgs_params)
        preds_adv = model(adv_x)
    elif args.attack == 'deepfool':
        deepfool_params = {'nb_candidate': 10, 'overshoot': 0.02, 'max_iterations': 100,
                           'clip_min': 0., 'clip_max': 1.}
        deepfool = attacks.DeepFool(wrapper, sess=sess)
        adv_x = deepfool.generate(x, **deepfool_params)
        preds_adv = model(adv_x)
    elif args.attack == 'cw':
        cw_params = {'batch_size': batch_size, 'binary_search_steps': 4, 'max_iterations': 1000,
                     'abort_early': True, 'clip_min': 0., 'clip_max': 1.}
        cw = attacks.CarliniWagnerL2(wrapper, sess=sess)
        adv_x = cw.generate(x, **cw_params)
        preds_adv = model(adv_x)
    elif args.attack == 'pgd':
        pgd_params = {'eps': eps, 'eps_iter': eps / 10, 'ord': order, 'clip_min': 0., 'clip_max': 1.}
        pgd = attacks.ProjectedGradientDescent(wrapper, sess=sess)
        adv_x = pgd.generate(x, **pgd_params)
        preds_adv = model(adv_x)
    # Evaluate the accuracy on adversarial examples
    '''
    if args.attack == 'cw':
        acc = utils_tf.model_eval(sess, x, y, preds_adv, x_test, y_test, args=eval_params)
        print('Test accuracy on adversarial examples: %0.4f' % acc)
    elif args.attack == 'lbfgs':
        acc = utils_tf.model_eval(sess, x, y, preds_adv, x_test, y_test, args=eval_params)
        print('Test accuracy on adversarial examples: %0.4f' % acc)
    else:
        evaluate(preds_adv, x_test, y_test, 'clean_train_adv_eval', True)
    '''
    
    with sess.as_default():
        adv = np.zeros(x_test.shape, dtype=np.float32)
        n = batch_size
        for i in range(x_test.shape[0] // n):
            adv[i * n:(i + 1) * n] = sess.run(adv_x, feed_dict={x: x_test[i * n:(i + 1) * n]})
            
    return adv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--attack", help="Attack of adversarial examples", default="fgsm")
    parser.add_argument("-e", "--eps", help="error tolerance", default=0.5)
    parser.add_argument("-i", "--input", help="input data path", default="../data")
    parser.add_argument("-o", "--output", help="output AE path", default="../outputs")
    parser.add_argument("-l", "--lorder", help="constraint distance order", default=2)
    parser.add_argument("-m", "--model", help="model path", default=None)
    args = parser.parse_args()

    if args.model is None:
        print("Please provide model path. Using -m or --model.")
        exit(0)
    if args.lorder == "inf":
        args.lorder = np.inf
    else:
        args.lorder = int(args.lorder)
    if args.lorder not in [0, 2, np.inf]:
        print("Please select correct constraint order. 0, 2, or inf.")
        exit(0)

    data = Dataset(args.input)
    start = time.clock()
    adv = adversarial_attack(data, args)
    elapsed = (time.clock() - start)
    print("Processing time:", elapsed)

    output_path = args.output
    np.save(os.path.join(output_path, "ae_{}_{}_linf_100.npy".format(args.attack, args.eps)), adv)


if __name__ == '__main__':
    main()
