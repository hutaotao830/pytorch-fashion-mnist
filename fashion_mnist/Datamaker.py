import os
from skimage import io
import torchvision.datasets.mnist as mnist

root = os.path.dirname(os.path.realpath(__file__))
train_set = (
    mnist.read_image_file(os.path.join(root, 'train-images-idx3-ubyte')),
    mnist.read_label_file(os.path.join(root, 'train-labels-idx1-ubyte'))
        )
test_set = (
    mnist.read_image_file(os.path.join(root, 't10k-images-idx3-ubyte')),
    mnist.read_label_file(os.path.join(root, 't10k-labels-idx1-ubyte'))
        )
print("training set :",train_set[0].size())
print("test set :",test_set[0].size())

def convert_to_img(train=True,debug=False):
    if(train):
        f=open(os.path.join(root,'train.txt'),'w')
        data_path=os.path.join(root,'train')
        if(not os.path.exists(data_path)):
            os.makedirs(data_path)
        for i, (img,label) in enumerate(zip(train_set[0],train_set[1])):
            img_path=os.path.join(data_path, str(i)+'.jpg')
            #io.imsave(img_path,img.numpy())
            f.write(img_path+' '+str(int(label))+'\n')
            if debug:
                break
        f.close()
    else:
        f = open(os.path.join(root, 'test.txt'), 'w')
        data_path = os.path.join(root, 'test')
        if (not os.path.exists(data_path)):
            os.makedirs(data_path)
        for i, (img,label) in enumerate(zip(test_set[0],test_set[1])):
            img_path = os.path.join(data_path, str(i) + '.jpg')
            #io.imsave(img_path, img.numpy())
            f.write(img_path + ' ' + str(int(label)) + '\n')
            if debug:
                break
        f.close()

if __name__ == '__main__':
    convert_to_img(True, debug=False)
    convert_to_img(False, debug=False)