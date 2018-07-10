---
 fashion mnist classification-[pytorch implementation]
 一个使用pytorch的完整例子
---
### 1. Details of file fold:
- fashion_mnist/
- fashion_mnist/train-images-idx3-ubyte.gz
- fashion_mnist/train-labels-idx1-ubyte.gz
- fashion_mnist/t10k-images-idx3-ubyte.gz
- fashion_mnist/t10k-labels-idx1-ubyte.gz
- fashion_mnist/mnist_reader.py
- fashion_mnist/Datamaker.py
- model/
- model/FashionNet.py
- FashionCnn.py
- MyFashionMnistCnn.pthx

### 2. File description:

| file | description|
|---|---|
|fashion_mnist/train-images-idx3-ubyte.gz|解压后得到train-images-idx3-ubyte|
|fashion_mnist/train-labels-idx1-ubyte.gz|解压后得到train-labels-idx1-ubyte|
|fashion_mnist/t10k-images-idx3-ubyte.gz|解压后得到t10k-images-idx3-ubyte|
|fashion_mnist/t10k-labels-idx1-ubyte.gz|解压后得到t10k-labels-idx1-ubyte|
|fashion_mnist/mnist_reader.py|读取.gz文件返回图片及对应标签|
|fashion_mnist/Datamaker.py|读取*-ubyte文件，生产图片目录及相应的标签文件|
|model/FashionNet.py|训练中使用cnn模型|
|FashionCnn.py|训练cnn模型，并保存训练好的模型MyFashionMnistCnn.pthx|
|inferrer.py|使用MyFashionMnistCnn.pthx模型预测给定图片|
|MyFashionMnistCnn.pthx|训练好的模型|

### 3. Running example:
requirements:
```python
PIL
torch
torchvision
```
running example:
```python
#读取*-ubyte文件，生产图片目录及相应的标签文件
python Datamaker.py
#重新训练cnn：
python FashionCnn.py
#使用训练好的模型：
python inferrer.py
```
output:
```python
epoch 1
Train Loss: 0.008292, Acc: 0.806033
Test Loss: 0.005567, Acc: 0.871100
epoch 2
Train Loss: 0.005097, Acc: 0.881100
Test Loss: 0.004908, Acc: 0.888100
epoch 3
Train Loss: 0.004356, Acc: 0.898083
Test Loss: 0.004504, Acc: 0.896100
epoch 4
Train Loss: 0.003861, Acc: 0.908233
Test Loss: 0.004241, Acc: 0.898200
epoch 5
Train Loss: 0.003514, Acc: 0.916700
Test Loss: 0.003992, Acc: 0.907100
epoch 6
Train Loss: 0.003208, Acc: 0.923400
Test Loss: 0.003937, Acc: 0.908700
epoch 7
Train Loss: 0.002930, Acc: 0.930800
Test Loss: 0.003899, Acc: 0.914800
epoch 8
Train Loss: 0.002710, Acc: 0.935700
Test Loss: 0.003765, Acc: 0.915900
epoch 9
Train Loss: 0.002472, Acc: 0.941767
Test Loss: 0.004039, Acc: 0.910600
epoch 10
Train Loss: 0.002257, Acc: 0.946667
Test Loss: 0.004064, Acc: 0.913400
```
### 4. Dataset:
We use the following dataset for our example:
[link](https://github.com/zalandoresearch/fashion-mnist).