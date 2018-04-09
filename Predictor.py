### Predictor 
%matplotlib inline
import matplotlib.pyplot as plt

print("Predictor Start")

# function to draw image
def draw_image(image_list, caption_list, batch_size, width, height):
    num_image = len(image_list)
    for i in range(num_image):
        image_list[i] = image_list[i].reshape((batch_size, width, height))

    for i in range(batch_size):
        fig = plt.figure(figsize=(20,20))
        for j in range(num_image):
            subplot = fig.add_subplot(1,num_image,j+1)
            subplot.set_xticks([])
            subplot.set_yticks([])
            subplot.set_title(caption_list[j]+str(i))
            # image_list[j] is each images with 4-D(first dimension is batch_size)
            subplot.imshow(image_list[j][i], vmin=0, vmax=255, cmap=plt.cm.gray, interpolation="nearest")

gan = GAN("predictList.csv", batch_size=1, is_data_augmentation=False)
generator = gan.generator
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(coord=coord, sess=gan.sess)

with gan.sess as sess:
    gan.saver.restore(sess, "./saved_session/s-100")
    
    try:
        while not coord.should_stop():
            image_in_csv, depth_data = gan.sess.run([generator.image_batch, generator.depth_batch])
            image_in_csv = image_in_csv.reshape((generator.batch_size, Generator.nPixels))

            out, x_input = gan.sess.run([generator.output, generator.x_image], feed_dict={generator.x:image_in_csv, generator.keep_prob:1.0})

            draw_image([x_input, out],
                       ["Input Image", "Output Image"],
                       generator.batch_size, generator.outputWidth, generator.outputHeight)
            coord.request_stop()

    except tf.errors.OutOfRangeError:
        coord.request_stop(e)
    finally:
        # When done, ask the threads to stop.
        coord.request_stop()

    # stop our queue threads and properly close the session
    coord.request_stop()
    coord.join(threads)

gan.sess.close()

print("Predictor End")
