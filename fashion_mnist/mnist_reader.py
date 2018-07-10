# -*- coding: utf-8 -*-
"""
Created on Tue Jul 10 19:48:18 2018

@author: hubinbin
"""

def load_mnist(kind='train',path=None):
    import os
    import gzip
    import numpy as np
    
    root = os.path.dirname(os.path.realpath(__file__))
    if path==None:
        path=root
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte.gz'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte.gz'
                               % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)

    return images, labels

if __name__ == '__main__':
    load_mnist()