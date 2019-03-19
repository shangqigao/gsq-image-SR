'''
Created on Jan 4, 2019

@author: gsq
'''
import os, shutil, time
import numpy as np
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tensorflow as tf
from PIL import Image
from scipy import misc
import matplotlib.pyplot as plt
import msdnn

def input_test(test_LR_dir, number):
    """
    Args;
        test_LR_dir: The directory of LR images
        test_HR_dir: The directory of HR images
        number: The index of one image
    Returns:
        test_image: a image will be tested, narray, float32
        test_label: a label, narray, float32
    """
    LR_names = os.listdir(test_LR_dir)
    LR_names.sort()
    if number < len(LR_names):
        LR_image = Image.open(test_LR_dir + '/' + LR_names[number])
        LR_img = np.asarray(LR_image, dtype='float32')
        return LR_img, LR_names[number]
    else:
        print('The input number is out of index')

def flip(image):
    images = [image]
    images.append(image[::-1, :, :])
    images.append(image[:, ::-1, :])
    images.append(image[::-1, ::-1, :])
    images = np.stack(images)
    return images
        
def mean_of_flipped(images):
    image = (images[0] + images[1, ::-1, :, :] + images[2, :, ::-1, :] +
             images[3, ::-1, ::-1, :])*0.25
    return image

def rotation(images):
    return np.swapaxes(images, 1, 2)

def main():
    cw_path = os.getcwd()
    test_LR_dir = os.path.join(cw_path, 'RealSR/Test_LR')
    fakeHR_path = os.path.join(cw_path, 'RealSR/Test_HR')
    if os.path.exists(fakeHR_path):
        shutil.rmtree(fakeHR_path)
    os.makedirs(fakeHR_path)
    ensemble = True
    per_time = []
    for number in range(20):
        start_time = time.time()
        image, LR_name = input_test(test_LR_dir, number)
        image_or = image
        d0 = image_or.shape[0] % 4
        d1 = image_or.shape[1] % 4
        lr_img = []
        sr_img = []
        lr_img.append(image_or[0:image_or.shape[0]-d0, 0:image_or.shape[1]-d1,:])
        lr_img.append(image_or[d0:image_or.shape[0], d1:image_or.shape[1], :])
        with tf.Graph().as_default():
            images_placeholder = tf.placeholder(tf.float32, shape=(None, None, None, 3))
            output = msdnn.infer_sr(images_placeholder)
            output = tf.clip_by_value(output, 0.0, 1.0)
            saver = tf.train.Saver()
            sess = tf.Session()
            saver.restore(sess, os.path.join(cw_path, 'model/msdnn_feed_v1_96a96_64blocks_1000000/model.checkpoint-999999'))
            for i in range(1):
                image = lr_img[i]
                if ensemble:
                    input_images = flip(image/255.0)
                    feed_dict1 = {images_placeholder : np.expand_dims(input_images[0,:,:,:], axis=0)}
                    image1 = sess.run(output, feed_dict=feed_dict1)
                    for j in range(3):
                        feed_dict1 = {images_placeholder : np.expand_dims(input_images[j+1,:,:,:],axis=0)}
                        image2 = sess.run(output, feed_dict=feed_dict1)
                        image1 = np.append(image1, image2, axis=0)
                    output_image1 = mean_of_flipped(image1)
                    feed_dict2 = {images_placeholder : np.expand_dims(rotation(input_images)[0,:,:,:], axis=0)}
                    image1 = sess.run(output, feed_dict=feed_dict2)
                    for j in range(3):
                        feed_dict2 = {images_placeholder : np.expand_dims(rotation(input_images)[j+1,:,:,:], axis=0)}
                        image2 = sess.run(output, feed_dict=feed_dict2)
                        image1 = np.append(image1, image2, axis=0)
                    output_image2 = mean_of_flipped(rotation(image1))
                    output_image = (output_image1 + output_image2)*0.5
                else:
                    input_images = np.expand_dims(image/255.0, 0)
                    feed_dict1 = {images_placeholder : input_images}
                    output_images = sess.run(output, feed_dict=feed_dict1)
                    output_image = output_images[0]
                sr_img.append(output_image)
            image_or[0:image_or.shape[0]-d0, 0:image_or.shape[1]-d1,:] = sr_img[0]*255.0
            fake_HR = np.around(image_or).astype(np.uint8)
            out_HR = Image.fromarray(fake_HR, 'RGB')
            out_HR.save(os.path.join(fakeHR_path, LR_name))
        duration = time.time() - start_time
        per_time.append(duration)
        print(number)
        print('\033[1;33;44m %s is successfully reconstructed! \033[0m' % LR_name)
    print('per time:', np.mean(np.asarray(per_time)))

        

if __name__ == '__main__':
    main()
