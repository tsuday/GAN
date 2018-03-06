## Trainer for GAN

import tensorflow as tf
import numpy as np
%matplotlib inline
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
import os
import sys

print("start")

# function to draw image
def draw_image(image_list, caption_list, batch_size, width, height):
    num_image = len(image_list)
    for i in range(num_image):
        image_list[i] = image_list[i].reshape((batch_size, width, height))

    for i in range(batch_size):
        fig = plt.figure(figsize=(15,15))
        for j in range(num_image):
            subplot = fig.add_subplot(1,num_image,j+1)
            subplot.set_xticks([])
            subplot.set_yticks([])
            subplot.set_title(caption_list[j]+str(i))
            # image_list[j] is each images with 4-D(first dimension is batch_size)
            subplot.imshow(image_list[j][i], vmin=0, vmax=255, cmap=plt.cm.gray, interpolation="nearest")

# Use AutoEncoder class as generator
with tf.Graph().as_default(): # both of generator and discriminator belongs to the same graph
    generator = AutoEncoder('dataList.csv', batch_size=8)
    discriminator = Discriminator(generator.x_image, generator.output, generator.t_compare)

    def generator_loss_function(output, target):
        eps = 1e-7
        loss_L1 = tf.reduce_mean(tf.abs(target-output))
        loss_discriminator = tf.reduce_mean(-tf.log(discriminator.layer_generator_output + eps))

        ratio_discriminator = 0.01
        return (1.00 - ratio_discriminator) * loss_L1 + ratio_discriminator * loss_discriminator

    generator.loss_function = generator_loss_function
    
    generator.sess.run(tf.global_variables_initializer())


#ae = AutoEncoder('dataList.csv', batch_size=4, is_skip_connection=True, is_data_augmentation=False, loss_function=AutoEncoder.L2)

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(coord=coord, sess=generator.sess)
scaler = MinMaxScaler(feature_range=(0,1))

# If you want to resume learning, please set start step number larger than 0
start = 0 #17200 #51600
# loop counter
i = start

# number of loops to report loss value while learning
n_report_loss_loop = 100#4# 200

# number of loops to report predicted image while learning
# this must be multiple number of n_report_loss_loop
n_report_image_loop = 400#20# 1000

# number of all loops for learning
n_all_loop = 2000000#42 #2000000


print("Start Training loop")
with generator.sess as sess:
    
    if start > 0:
        print("Resume from session files")
        generator.saver.restore(sess, "./saved_session/s-" +str(start))
    
    try:
        while not coord.should_stop():
            i += 1
            # Run training steps or whatever
            image_data, depth_data = generator.sess.run([generator.image_batch, generator.depth_batch])
            image_data = image_data.reshape((generator.batch_size, AutoEncoder.nPixels))
            #image_data = scaler.fit_transform(image_data)
            depth_data = depth_data.reshape((generator.batch_size, AutoEncoder.nPixels))
            
            generator.sess.run([generator.train_step], feed_dict={generator.x:image_data, generator.t:depth_data, generator.keep_prob:0.5})
            if i == n_all_loop:
                coord.request_stop()

            #TODO:Split data into groups for cross-validation
            if i==start+1 or i % n_report_loss_loop == 0:
                loss_vals = []
                loss_val, t_cmp, out, summary, x_input, discriminator_loss = generator.sess.run([generator.loss, generator.t_compare, generator.output, generator.summary, generator.x_image, discriminator.loss],
                                                            feed_dict={generator.x:image_data, generator.t:depth_data, generator.keep_prob:1.0})
                loss_vals.append(loss_val)
                loss_val = np.sum(loss_vals)

                # TODO: save for discriminator
                generator.saver.save(generator.sess, './saved_session/s', global_step=i)
                
                print ('Step: %d, Loss: %f @ %s' % (i, loss_val, datetime.now().strftime("%Y/%m/%d %H:%M:%S")))
                if i==start+1 or i % n_report_image_loop == 0:
                    x_input = tf.reshape(x_input, [generator.batch_size, generator.outputWidth, generator.outputHeight])
                    t_cmp = tf.reshape(t_cmp, [generator.batch_size, generator.outputWidth, generator.outputHeight])
                    out = tf.reshape(out, [generator.batch_size, generator.outputWidth, generator.outputHeight])
                    draw_image([x_input.eval(session=generator.sess), out.eval(session=generator.sess), t_cmp.eval(session=generator.sess)],
                               ["Input Image", "Predicted Result", "Ground Truth"],
                               generator.batch_size, generator.outputWidth, generator.outputHeight)
                    
                    # TODO:write for discriminator
                    generator.writer.add_summary(summary, i)

    except tf.errors.OutOfRangeError as e:
        print('Done training')
        coord.request_stop(e)
    except Exception as e:
        print('Caught exception')
        print(sys.exc_info())
        coord.request_stop(e)
    finally:
        # When done, ask the threads to stop.
        coord.request_stop()

    # stop our queue threads and properly close the session
    coord.request_stop()
    coord.join(threads)

generator.sess.close()

print("end")
