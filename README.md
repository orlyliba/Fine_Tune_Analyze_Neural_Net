# Fine tune and analyze deep-learning neural networks
Gain some experience in deep-learning by fine-tuning and analyzing AlexNet with TensorFlow.

This is a beginner-level project that starts with a pre-trained neural network and then fine-tunes it for a task of your choosing. There is also code for anlyzing the network, such as: visualizing the weights and layer activations, occulusion analysis, t-SNE and image with highest probability in class.

Recommended reading:
- Deep-learning Stanford class, [CS231n](http://cs231n.github.io/)
- [AlexNet paper] (https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)
- ["Visualizing and Understanding Convolutional Networks" paper] (https://arxiv.org/pdf/1311.2901v3.pdf)

Code and weights for AlexNet in TensorFlow are from [here]( http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/) (Thank you!)

Recommended steps:

0. Install [python](https://www.python.org/downloads/), [Tensorflow](https://www.tensorflow.org/get_started/os_setup) and [Jupyter Notebook](http://jupyter.readthedocs.io/en/latest/install.html). I've found that no matter what I did, Tensorflow did not work on Windows (as of Oct. 2016), therefore I worked with Linux. I also recommend looking at other libraries for deep learning (such as Caffe, Torch, Theano...), because Tensorflow is not the only option and it's not perfect for every use. That said, TensorFlow is great for beginners who know python.
1. Run AlexNet pre-trained on ImageNet, just to test out the classification.
2. Visualize the behaviour of the network: look at the weights and activations of the different layers
3. Define the next task you would like the network to work on. As an example, I deined a task for classifying between dogs, cats and flowers. This code will download images based on a Google search. Then you'll need to crop or resize them to match the size expected by AlexNet (227x227x3) using this code. This example distinguishes between three classes, and you can change the number of classes in your task by updating the fc8 layer in the network.
4. Fine-tune AlexNet. You can define or loop over different values of learning rates, drop-out, tuning depth (which layers will be fine-tuned), etc.
5. Save your network
6. Find the images with the highest probability in each class
7. Visualize your network's behavior by plotting your images on a scatter plot using [t-SNE](https://en.wikipedia.org/wiki/T-distributed_stochastic_neighbor_embedding)
8. Occlusion analysis is a nice way of visializing which regions in your image are significant for your your network's decision




