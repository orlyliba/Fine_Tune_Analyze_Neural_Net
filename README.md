# Fine tune and analyze deep-learning neural networks
Fine-tune and analyze a deep neural network for visual recognition (AlexNet) with TensorFlow.

This is a beginner-level project that starts with a pre-trained neural network and then fine-tunes it for a task of your choosing. There is also code for anlyzing the network, such as: visualizing the weights and layer activations, occulusion analysis, t-SNE and image with highest probability in class.

Code and weights for AlexNet in TensorFlow are from [here]( http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/) (Thank you!)

### Recommended steps: 

0. Install [Python](https://www.python.org/downloads/) (+ numpy, matplotlib, scipy, sklearn...), [Tensorflow](https://www.tensorflow.org/get_started/os_setup) and [Jupyter Notebook](http://jupyter.readthedocs.io/en/latest/install.html). I've found that no matter what I did, Tensorflow did not work on Windows (as of Oct. 2016), therefore I used Linux (Ubuntu). I also recommend looking at other libraries for deep learning (such as Caffe, Torch, Theano...), because Tensorflow is not the only option and it's not perfect for every use. That said, TensorFlow is great for beginners who use Python.
1. Run AlexNet pre-trained on ImageNet (weights and biases can be found [here](http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/bvlc_alexnet.npy)), and test out the classification ([link](https://github.com/orlyliba/Fine_Tune_Analyze_Neural_Net/blob/master/AlexNet_notebook.ipynb)).
2. Visualize the behaviour of the network: look at the weights and activations of the different layers ([link](https://github.com/orlyliba/Fine_Tune_Analyze_Neural_Net/blob/master/AlexNet_vis_notebook.ipynb)).
3. Define the next task you would like the network to work on. As an example, I defined a task for classifying between dogs, cats and flowers. [This](https://github.com/orlyliba/Fine_Tune_Analyze_Neural_Net/blob/master/image_dl.py) code will download images based on a Google search. Then you'll need to crop or resize them to match the size expected by AlexNet (227x227x3) using [this](https://github.com/orlyliba/Fine_Tune_Analyze_Neural_Net/blob/master/Prep_Images.ipynb) code. This project distinguishes between three classes, and you can change the number of classes in your task by updating the size of the fc8 layer in the network (search for n_classes).
4. Fine-tune AlexNet. You can define or loop over different values of learning rates, drop-out, tuning depth (which layers will be fine-tuned), etc ([link](https://github.com/orlyliba/Fine_Tune_Analyze_Neural_Net/blob/master/AlexNet_finetune_select_layers.ipynb)).
5. Save your network ([link](https://github.com/orlyliba/Fine_Tune_Analyze_Neural_Net/blob/master/AlexNet_finetune_save_net.ipynb)).
6. Find the images with the highest probability in each class ([link](https://github.com/orlyliba/Fine_Tune_Analyze_Neural_Net/blob/master/Find_highest_prob_image.ipynb)).
7. Visualize your network's behavior by plotting your images on a scatter plot using [t-SNE](https://en.wikipedia.org/wiki/T-distributed_stochastic_neighbor_embedding) ([link](https://github.com/orlyliba/Fine_Tune_Analyze_Neural_Net/blob/master/AlexNet_TSNE.ipynb)). This was done in collaboration with [Tanya Glozman](http://tanyaglozman.wixsite.com/aboutme).
8. Occlusion analysis is a nice way of visializing which regions in your image are significant for your network's decision ([link](https://github.com/orlyliba/Fine_Tune_Analyze_Neural_Net/blob/master/Occlusion_analysis.ipynb)).


### Occlusion analysis 
Probability map for correctly classifying the cat, as function of the location of an occluding square in the image. The probability for classifying the cat correctly reduces when the cat's face is occluded.
![](https://cloud.githubusercontent.com/assets/19598320/21302263/50be3440-c56a-11e6-9302-aa76cb52eeec.png)

### t-SNE visualization 
A 2-dimensional representation of the output of the network, showing the separation between images of different classes (cats, dogs and flowers).
![](https://cloud.githubusercontent.com/assets/19598320/21300571/f575930e-c559-11e6-9324-50ca4d98c09d.png)

### Recommended reading:
- Deep-learning Stanford class, [CS231n](http://cs231n.github.io/)
- [AlexNet paper] (https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)
- ["Visualizing and Understanding Convolutional Networks" paper] (https://arxiv.org/pdf/1311.2901v3.pdf)

