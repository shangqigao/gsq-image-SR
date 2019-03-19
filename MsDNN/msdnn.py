'''
Created on Jan 4, 2019

@author: gsq
'''
import tensorflow as tf
import numpy as np
#The size of extracted sub-images, which is set by user.
label_size = 64
upscale = 1
IMAGE_SIZE = label_size // upscale
#LAST_DIMNESION is for 'features', it would be 1 for grayscale, 3 for RGB images, 4 for RGBA, etc. 
LAST_DIMENSION = 3
IMAGE_PIXELS = IMAGE_SIZE*IMAGE_SIZE*LAST_DIMENSION
#The size of hidden filters, it would be 9 for hidden 1, 1 for hidden 2, 5 for output layer.
hidden_filter = [3, 3, 3]
hidden_units0 = [12, 12, 12]
hidden_units2 = [96, 96, 96]
hidden_units4 = [96, 96, 96]
hidden_units8 = [64, 256, 64]
output_filter = 3
skip_filter = 5
num_blocks = 64
LABEL_PIXELS = IMAGE_PIXELS*upscale*upscale
mean = np.asarray([0.4509, 0.4239, 0.3875], dtype=np.float32)
            
def res_block(image, size, units):
    identity = image
    #The first convolutional layer
    image = tf.layers.conv2d(image,
                             filters=units[1], kernel_size=size[1], strides=1, 
                             padding='SAME', use_bias=True, name='conv0')
    
    image = tf.nn.relu(image)
    
    image = tf.layers.conv2d(image, 
                             filters=units[2], kernel_size=size[2], strides=1, 
                             padding='SAME', use_bias=True, name='conv1')

    return image + identity


#Define the sub-pixel shuffle block
def subpixel_block(LR_img, size, upscale):
    LR_img = tf.layers.conv2d(LR_img,
                              filters=3*upscale*upscale, kernel_size=size, strides=1,
                              padding='SAME', use_bias=True, name='conv0')
    LR_img = tf.depth_to_space(LR_img, upscale)
    return LR_img
    

def infer_sr(img_batch):
    """ Build the MsDNN model for inference
    
    Args:
        img_batch: Images placeholder, from input low-resolution images
    
    Returns:
        Convolutional neural network: Output high-resolution images 
    
    """
    img_batch -= mean
    img_batch1 = img_batch
    img_batch2 = img_batch
    #No downscale
    with tf.variable_scope('down0'):
        img_batch = tf.layers.conv2d(img_batch,
                                filters=3, kernel_size=hidden_filter[0], strides=1,
                                padding='SAME', use_bias=True, name='conv0')
    #downscale=2
    with tf.variable_scope('d2input'):
        img_batch1 = tf.layers.conv2d(img_batch1,
                                      filters=hidden_units2[0], kernel_size=hidden_filter[0], strides=2,
                                      padding='SAME', use_bias=True, name='conv0')

    for num in range(num_blocks):
        with tf.variable_scope('d2block{}'.format(num)):
            img_batch1 = res_block(img_batch1, hidden_filter, hidden_units2)

    with tf.variable_scope('d2output'):
        img_batch1 = subpixel_block(img_batch1, output_filter, upscale*2)
    #downscale=4
    with tf.variable_scope('d4input'):
        img_batch2 = tf.layers.conv2d(img_batch2, 
                                      filters=hidden_units4[0]//2, kernel_size=hidden_filter[0], strides=2,
                                      padding='SAME', use_bias=True, name='conv0')
        img_batch2 = tf.layers.conv2d(img_batch2,
                                      filters=hidden_units4[0], kernel_size=hidden_filter[0], strides=2,
                                      padding='SAME', use_bias=True, name='conv1')
    for num in range(num_blocks):
        with tf.variable_scope('d4block{}'.format(num)):
            img_batch2 = res_block(img_batch2, hidden_filter, hidden_units4)

    with tf.variable_scope('d4output'):
        img_batch2 = subpixel_block(img_batch2, output_filter, upscale*4)
    #concatenate
    img_batch = tf.concat([img_batch, img_batch1, img_batch2], axis=3)
    with tf.variable_scope('output'):
        img_batch = tf.layers.conv2d(img_batch,
                                     filters=3, kernel_size=hidden_filter[0], strides=1,
                                     padding='SAME', use_bias=True, name='conv0')
    return img_batch + mean      
         
           
def loss(output, labels):
    """Calculate the loss from output and labels
            
    Args:
        output: Reconstructed tensor, float, --[batch_size, LABEL_PIXELS]
        labels: Labels tensor, float, --[batch_size, LABEL_PIXELS]
                
    Returns:
        loss: Loss tensor of type of float.
            
    """
    mean_squared_error = tf.losses.absolute_difference(labels, output)
    return mean_squared_error


def training(loss, learning_rate):
    """Set up the training operations
    Create a summarizer to track the loss over time in TensorBoard.
    Create an optimizer and apply gradients to all trainable variables.
    
    The operation returned by this function is what must be passed to
    "sess.run()" call to cause the model to train
    
    Args:
        loss: L_2 loss tensor from loss()
        learning_rate: The learning rate to use for gradient descent.
        
    Returns:
        train_op: the operation for training.
    
    """
    #Add a scalar summary for the L2 loss.
    tf.summary.scalar('loss', loss)
    
    #Create the gradient descent optimizer with the given learning rate.
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    
    #Create a variable to track the global step.
    global_step = tf.Variable(0, name='global_step', trainable=False)
    
    # Use the optimizer to apply the gradients that minimize the loss
    # (and also increment the global step counter) as a single training step.
    train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op

def evaluation(output, labels):
    """Evaluate the quality of the output at reconstructing the label.
    
    Args:
        output: Reconstructed tensor, float, --[batch_size, IMAGE_PIXELS]
        labels: Lables tensor, float, --[batch_size, IMAGE_PIXELS]
        
    Returns: PSNR and SSIM
        
    """
    #Calculate psnr and ssim
    psnr = tf.reduce_mean(tf.image.psnr(output, labels, max_val=1.0))
    ssim = tf.reduce_mean(tf.image.ssim(output, labels, max_val=1.0))
    return psnr, ssim
