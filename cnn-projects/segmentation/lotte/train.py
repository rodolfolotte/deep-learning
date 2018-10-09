import os, sys
import json
import imp
import time
import tensorflow as tf
import logging
import string
import numpy as np

import dl.autoencoder as autoencoder
import inputs.prepare as inputs
import evaluate.eval as evaluation
import optimizer.optimizer as optimizer
import inference.inference as inference

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.INFO, stream=sys.stdout)    

def trainingSettings(hypes):   
    learning_rate = tf.placeholder(tf.float32)
    
    logging.info("...Creating the queues from processing")
    with tf.name_scope("Queues"):
        queue = inputs.createQueues(hypes, 'train')

    logging.info("...Add Input producers to the graph")
    with tf.name_scope("Inputs"):
        image, labels = inputs.inputs(hypes, queue, phase='train')

    logging.info("...Build the encoder using the VGG16 model, with 13 convolutional layers")
    encoder_logits = autoencoder.buildEncoder(hypes, image, train=True)

    logging.info("...Build decoder with 3 transposed layers")
    decoded_logits = autoencoder.buildDecoder(hypes, encoder_logits, train=True)

    logging.info("...Add the optimizer to the graph, for loss calculation")
    with tf.name_scope("Loss"):
        losses = autoencoder.loss(hypes, decoded_logits, labels)

    logging.info("...Add the optimizer to the graph, which calculate and apply gradients")
    with tf.name_scope("Optimizer"):        
        global_step = tf.Variable(0, trainable=False)
        train_op = optimizer.training(hypes, losses, global_step, learning_rate)
        
    logging.info("...Evaluating the graph")
    with tf.name_scope("Evaluation"):        
        eval_list = autoencoder.evaluation(hypes, image, labels, decoded_logits, losses, global_step)
        summary_op = tf.summary.merge_all()

    graph = {}
    graph['losses'] = losses
    graph['eval_list'] = eval_list
    graph['summary_op'] = summary_op
    graph['train_op'] = train_op
    graph['global_step'] = global_step
    graph['learning_rate'] = learning_rate
    graph['decoded_logits'] = learning_rate
        
    # tf.contrib.layers.summarize_collection(tf.GraphKeys.WEIGHTS)
    # tf.contrib.layers.summarize_collection(tf.GraphKeys.BIASES)
    # summary_op = tf.summary.merge_all()    

    saver = tf.train.Saver(max_to_keep=10)
    sess = tf.get_default_session()
        
    if 'init_function' in hypes:
        _initalize_variables = hypes['init_function']
        _initalize_variables(hypes)
    else:
        init = tf.global_variables_initializer()
        sess.run(init)
        
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    summary_writer = tf.summary.FileWriter(hypes['data']['output_dir'], graph=sess.graph)
        
    session = {}
    session['sess'] = sess
    session['saver'] = saver
    session['summary_op'] = summary_op
    session['writer'] = summary_writer
    session['coord'] = coord
    session['threads'] = threads

    with tf.name_scope('Validation'):
        tf.get_variable_scope().reuse_variables()
        image_pl = tf.placeholder(tf.float32)
        image = tf.expand_dims(image_pl, 0)
        image.set_shape([1, None, None, 3])
        inference_graph = inference.buildInferenceGraph(hypes, image)
        graph['image_pl'] = image_pl
        graph['inf_out'] = inference_graph
    
    return graph, session, queue


# def print_training_status(hypes, step, loss_value, start_time, lr):
#     info_str = 'Step {step}/{total_steps}: loss = {loss_value:.2f}; lr = {lr_value:.2e}; {sec_per_batch:.3f} sec (per Batch); {examples_per_sec:.1f} imgs/sec' 
#     duration = (time.time() - start_time) / 50
#     examples_per_sec = hypes['solver']['batch_size'] / duration
#     sec_per_batch = float(duration)
#     logging.info(info_str.format(step=step,
#                                  total_steps=hypes['solver']['max_steps'],
#                                  loss_value=loss_value,
#                                  lr_value=lr,
#                                  sec_per_batch=sec_per_batch,
#                                  examples_per_sec=examples_per_sec))

def print_training_status(hypes, step, start_time, lr):
    info_str = 'Step {step}/{total_steps}: lr = {lr_value:.2e}; {sec_per_batch:.3f} sec (per Batch); {examples_per_sec:.1f} imgs/sec' 
    duration = (time.time() - start_time) / 50
    examples_per_sec = hypes['solver']['batch_size'] / duration
    sec_per_batch = float(duration)
    logging.info(info_str.format(step=step,
                                 total_steps=hypes['solver']['max_steps'],                                 
                                 lr_value=lr,
                                 sec_per_batch=sec_per_batch,
                                 examples_per_sec=examples_per_sec))

def print_eval_dict(eval_names, eval_results):
    print_str = string.join([nam + ": %.2f" for nam in eval_names], ', ')
    print_str = "   " + print_str
    logging.info(print_str % tuple(eval_results))


def runTraining(hypes, graph, session):    
    # summary = tf.Summary()
    
    sess = session['sess']
    # summary_writer = session['writer']

    display_iter = hypes['epochs']['display_iter']
    write_iter = hypes['epochs'].get('write_iter', 5 * display_iter)
    eval_iter = hypes['epochs']['eval_iter']
    save_iter = hypes['epochs']['save_iter']
    image_iter = hypes['epochs'].get('image_iter', 5 * save_iter)

    # py_smoother = autoencoder.MedianSmoother(20)
    dict_smoother = autoencoder.ExpoSmoother(0.95)    
    eval_names, eval_ops = zip(*graph['eval_list'])

    logging.info("..Starting the training step...")
    start_time = time.time()

    for step in xrange(0, hypes['solver']['max_steps']):        
        lr = optimizer.getLearningRate(hypes, step)       
        feed_dict = {graph['learning_rate']: lr}     

        if step % display_iter:        
            sess.run([graph['train_op']], feed_dict=feed_dict)        
        elif step % display_iter == 0:                        
            print(step)
            sess.run([graph['train_op'], graph['losses']['total_loss']], feed_dict=feed_dict)
            
            print_training_status(hypes, step, start_time, lr)            
            eval_results = sess.run(eval_ops, feed_dict=feed_dict)            
            print_eval_dict(eval_names, eval_results)            
            dict_smoother.update_weights(eval_results)     
            # smoothed_results = dict_smoother.get_weights()
            # print_eval_dict(eval_names, smoothed_results, prefix='(smooth)')
            start_time = time.time()
        
        # if step % write_iter == 0:            
            # if FLAGS.summary:
            #     summary_str = sess.run(tfsess['summary_op'], feed_dict=feed_dict)
            #     summary_writer.add_summary(summary_str, global_step=step)
            # summary.value.add(tag='training/total_loss', simple_value=float(loss_value))
            # summary.value.add(tag='training/learning_rate', simple_value=lr)
            # summary_writer.add_summary(summary, step)
            # eval_results = np.array(eval_results)
            # eval_results = eval_results.tolist()
            # eval_dict = zip(eval_names, eval_results)
            # _write_eval_dict_to_summary(eval_dict, 'Eval', summary_writer, step)
            # eval_dict = zip(eval_names, smoothed_results)
            # _write_eval_dict_to_summary(eval_dict, 'Eval/smooth', summary_writer, step)

        # if step % eval_iter == 0 and step > 0 or (step + 1) == hypes['solver']['max_steps']:            
        #     logging.info('..Running Evaluation Script')
        #     eval_dict, images = evaluation.evaluate(hypes, sess, graph['image_pl'], graph['inference_out'])

            # logging.info("..Evaluation Finished. Saving images in " + hypes['data']['output_dir'])              
            # _write_images_to_summary(images, summary_writer, step)     

            # if images is not None and len(images) > 0:
            #    name = str(n % 10) + '_' + images[0][0]
            #    image_file = os.path.join(hypes['data']['image_dir'], name)
            #    scp.misc.imsave(image_file, images[0][1])
            #    n = n + 1

            # logging.info('.. Results:')
            # utils.print_eval_dict(eval_dict, prefix='(raw)')
            # _write_eval_dict_to_summary(eval_dict, 'Evaluation', summary_writer, step)

            # logging.info('..Smooth Results:')
            # names, res = zip(*eval_dict)
            # smoothed = py_smoother.update_weights(res)
            # eval_dict = zip(names, smoothed)
            # utils.print_eval_dict(eval_dict, prefix='(smooth)')
            # _write_eval_dict_to_summary(eval_dict, 'Evaluation/smoothed', summary_writer, step)
            # start_time = time.time()

        if step % save_iter == 0 and step > 0 or (step + 1) == hypes['solver']['max_steps']:
            logging.info('..Writing checkpoint: ' + hypes['data']['output_dir'])
            checkpoint_path = os.path.join(hypes['data']['output_dir'], 'model.ckpt')
            session['saver'].save(sess, checkpoint_path, global_step=step)            
            start_time = time.time()

        # if step % image_iter == 0 and step > 0 or (step + 1) == hypes['solver']['max_steps']:
        #     _write_images_to_disk(hypes, images, step)
    

def main():    
    with open(sys.argv[1]) as hypesFile:
        logging.info("Opening the neural model file settings (hyperparameters): %s", hypesFile)
        hypes = json.load(hypesFile)

    with tf.Session() as tfsess:
        logging.info("Setting up the autoencoder...")
        graph, sess, queue = trainingSettings(hypes)
        
        inputs.startEnqueuingThreads(hypes, queue, 'train', tfsess)

        logging.info("Initializing the training")
        runTraining(hypes, graph, sess)

if __name__ == "__main__":        
    main()