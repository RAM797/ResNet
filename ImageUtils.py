import numpy as np

""" This script implements the functions for data augmentation and preprocessing.
"""

def parse_record(record, training):
    """ Parse a record to an image and perform data preprocessing.

    Args:
        record: An array of shape [3072,]. One row of the x_* matrix.
        training: A boolean. Determine whether it is in training mode.

    Returns:
        image: An array of shape [3, 32, 32].
    """
    # Reshape from [depth * height * width] to [depth, height, width].
    depth_major = record.reshape((3, 32, 32))

    # Convert from [depth, height, width] to [height, width, depth]
    image = np.transpose(depth_major, [1, 2, 0])

    image = preprocess_image(image, training)

    # Convert from [height, width, depth] to [depth, height, width]
    image = np.transpose(image, [2, 0, 1])

    return image

def preprocess_image(image, training):
    """ Preprocess a single image of shape [height, width, depth].

    Args:
        image: An array of shape [32, 32, 3].
        training: A boolean. Determine whether it is in training mode.
    
    Returns:
        image: An array of shape [32, 32, 3].
    """
    if training:
        ### YOUR CODE HERE
        # Resize the image to add four extra pixels on each side.
        
        ### YOUR CODE HERE
        # stack_layers = [np.pad(image[:,:,ch], ((4,4),(4,4)),'constant', constant_values=255) for ch in range(3)]
        stack_layers = []
        for i in range(3):
            stack_layers.append(np.pad(image[:,:,i], ((4,4),(4,4)),'constant', constant_values=255))
        image = np.stack(stack_layers, axis=2)
        ### YOUR CODE HERE
        # Randomly crop a [32, 32] section of the image.
        # HINT: randomly generate the upper left point of the image
        
        ### YOUR CODE HERE
        upper_left = np.random.randint(0,9, (2,1))
        image = image[int(upper_left[0]):int(upper_left[0])+32, int(upper_left[1]):int(upper_left[1])+32,:]
        ### YOUR CODE HERE
        # Randomly flip the image horizontally.
        image = image if np.random.randint(0,50) > 25 else np.fliplr(image)
        ### YOUR CODE HERE

    ### YOUR CODE HERE
    # Subtract off the mean and divide by the standard deviation of the pixels.
    image = (image - np.mean(image))/np.std(image)
    ### YOUR CODE HERE

    return image