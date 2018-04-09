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

# Prepare network for GAN
gan = GAN('dataList.csv', batch_size=8)
generator = gan.generator
discriminator = gan.discriminator

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(coord=coord, sess=gan.sess)
scaler = MinMaxScaler(feature_range=(0,1))

# If you want to resume learning, please set start step number larger than 0
start = 0
# loop counter
i = start

# number of loops to report loss value while learning
n_report_loss_loop = 100 #4

# number of loops to report predicted image while learning
# this must be multiple number of n_report_loss_loop
n_report_image_loop = 300 #20

# number of all loops for learning
n_all_loop = 2000000 #42


print("Start Training loop")
with gan.sess as sess:
    
    if start > 0:
        print("Resume from session files")
        gan.saver.restore(sess, "./saved_session/s-" +str(start))
    
    try:
        while not coord.should_stop():
            i += 1
            # Run training steps or whatever
            image_data, depth_data = gan.sess.run([generator.image_batch, generator.depth_batch])
            image_data = image_data.reshape((generator.batch_size, Generator.nPixels))
            #image_data = scaler.fit_transform(image_data)
            depth_data = depth_data.reshape((generator.batch_size, Generator.nPixels))
            
            gan.sess.run([generator.train_step], feed_dict={generator.x:image_data, generator.t:depth_data, generator.keep_prob:0.5})
            if i == n_all_loop:
                coord.request_stop()

            if i==start+1 or i % n_report_loss_loop == 0:
                loss_vals = []
                loss_val, t_cmp, out, summary, x_input, discriminator_loss = gan.sess.run([generator.loss, generator.t_compare, generator.output, gan.summary, generator.x_image, discriminator.loss],
                                                            feed_dict={generator.x:image_data, generator.t:depth_data, generator.keep_prob:1.0})
                loss_vals.append(loss_val)
                loss_val = np.sum(loss_vals)

                gan.saver.save(gan.sess, './saved_session/s', global_step=i)
                
                print ('Step: %d, Loss: %f @ %s' % (i, loss_val, datetime.now().strftime("%Y/%m/%d %H:%M:%S")))
                if i==start+1 or i % n_report_image_loop == 0:
                    x_input = tf.reshape(x_input, [generator.batch_size, generator.outputWidth, generator.outputHeight])
                    t_cmp = tf.reshape(t_cmp, [generator.batch_size, generator.outputWidth, generator.outputHeight])
                    out = tf.reshape(out, [generator.batch_size, generator.outputWidth, generator.outputHeight])
                    draw_image([x_input.eval(session=gan.sess), out.eval(session=gan.sess), t_cmp.eval(session=gan.sess)],
                               ["Input Image", "Predicted Result", "Ground Truth"],
                               generator.batch_size, generator.outputWidth, generator.outputHeight)
                    
                    gan.writer.add_summary(summary, i)

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

gan.sess.close()

print("end")
