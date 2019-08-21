import os
import tensorflow as tf
import logging
logging.disable(logging.WARNING)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from models.modules import PatchGanDiscriminator, Generator212

class CycleGAN2(object):

    def __init__(self, num_features, batch_size=1, discriminator=PatchGanDiscriminator, generator=Generator212,
                 mode='train', log_dir='./log'):

        self.num_features = num_features
        self.batch_size = batch_size
        self.input_shape = [None, num_features, None]  # [batch_size, num_features, num_frames]

        self.discriminator = discriminator
        self.generator = generator
        self.mode = mode

        self.build_model()

        self.saver = tf.train.Saver(max_to_keep=100)
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())


    def build_model(self):

        # Placeholders for real training samples
        self.input_A_real = tf.placeholder(tf.float32, shape=self.input_shape, name='input_A_real')


        self.generation_B = self.generator(inputs=self.input_A_real, dim=self.num_features, batch_size=self.batch_size,
                                           reuse=False,
                                           scope_name='generator_A2B')


    def test(self, inputs, direction):
        if direction == 'A2B':
            # --------- Load frozen graph -------------------------
            #print('Loading model...')

            with tf.gfile.GFile("/Users/RanyaJo/GAN-Voice-Conversion/models/log/freeze_conv.pb", 'rb') as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())

            tf.import_graph_def(graph_def, {'input_A_real': self.input_A_real})
            #print('Model loading complete!')

            softmax_tensor = self.sess.graph.get_tensor_by_name("import/generator_A2B/conv_out/Conv2D:0")
            generation = self.sess.run(softmax_tensor, feed_dict={self.input_A_real: inputs})
            generation = tf.squeeze(generation, axis=[-1])

            sess = tf.InteractiveSession()
            with sess.as_default():
                generation = generation.eval()

        else:
            generation = self.sess.run(self.generation_A_test, feed_dict={self.input_B_test: inputs})
            raise Exception('Conversion direction must be specified.')

        return generation


    def save(self, directory, filename):

        if not os.path.exists(directory):
            os.makedirs(directory)
        self.saver.save(self.sess, os.path.join(directory, filename))

        return os.path.join(directory, filename)

    def load(self, filepath):

        self.saver.restore(self.sess, filepath)

if __name__ == '__main__':
    import numpy as np

    batch_size = 5
    model = CycleGAN2(num_features=36, batch_size=batch_size)
    gen = model.test(inputs=np.random.randn(batch_size, 36, 317), direction='A2B')
    print(gen.shape)
    print('Graph Compile Successeded.')