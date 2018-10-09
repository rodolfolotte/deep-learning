import sys
import logging
import tensorflow as tf

import dl.autoencoder as autoencoder

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.INFO, stream=sys.stdout)  

def buildInferenceGraph(hypes, image):
    logging.info("...Validating dataset")

    with tf.name_scope("Validation"):
        logits = autoencoder.buildEncoder(hypes, image, train=False)
        decoded_logits = autoencoder.buildDecoder(hypes, logits, train=False)

    return decoded_logits