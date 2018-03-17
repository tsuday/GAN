# GAN
import tensorflow as tf

# GAN(Generative Adversarial Network) class
# This class provides an interface between GAN user (training or prediction) and internal network (generator and discriminator).
class GAN:
    
    def __init__(self, training_csv_file_name, **options):
        ## options by argument
        self.batch_size = options.get('batch_size', 1)
        self.is_data_augmentation = options.get('is_data_augmentation', True)
        # Option for generator to skip conecctions between corresponding layers of encoder and decoder as in U-net
        self.is_skip_connection = options.get('is_skip_connection', True)
        self.loss_function = options.get('loss_function', Generator.L1)
        
        isDebug = True
        if isDebug:
            print("batch_size : {0}".format(self.batch_size))
            print("data_augmentation : {0}".format(self.is_data_augmentation))
            print("skip_connection : {0}".format(self.is_skip_connection))
            print("loss_function : {0}".format(self.loss_function))
        
        with tf.Graph().as_default(): # both of generator and discriminator belongs to the same graph
            print(training_csv_file_name)
            print(options)
            generator = Generator(training_csv_file_name, options)
            discriminator = Discriminator(generator.x_image, generator.output, generator.t_compare, options)

            def generator_loss_function(output, target):
                eps = 1e-7
                loss_L1 = tf.reduce_mean(tf.abs(target-output))
                loss_discriminator = tf.reduce_mean(-tf.log(discriminator.layer_generator_output + eps))

                ratio_discriminator = 0.01
                return (1.00 - ratio_discriminator) * loss_L1 + ratio_discriminator * loss_discriminator

            generator.loss_function = generator_loss_function
            generator.sess.run(tf.global_variables_initializer())
            
            self.generator = generator
            self.discriminator = discriminator
