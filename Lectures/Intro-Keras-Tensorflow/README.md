# Introduction to Keras and Tensorflow

The lesson culminates in coding up a first ANN in `keras`, but first there are some loose ends to tie up regarding backpropagation, batches, and optimizers.

## Learning Goals

- Describe the concept of backpropagation
- Explain the use of gradient descent in neural networks
- Use `keras` to code up a neural network model

## Lesson Materials

Two notebooks:
- [Jupyter Notebook: Intro to Keras and Tensorflow](intro_to_keras_and_tf.ipynb)
- [Jupyter Notebook: Neural Network from Scratch](neural_network_from_scratch.ipynb)

**Note from Greg:** the NN from Scratch notebook is in both this lecture and [the previous one on neural network architecture](../NNArchitecture). Current plan is to go through the first part in the previous session and the second part in this one.

## Lesson Plan 

### Introduction (5 Mins)

Setting up binary classification problem (odd or even) with digit data.

### Backpropagation (5 Mins)

Strictly, this is a matter of computing the gradient and not of using the gradient to adjust weights.

### Networks as Function Approximators (10 Mins)

The book chapter on how networks can be used to approximate any function (link in ntbk) is a nice illustration of how powerful these models are. There are widgets that can be adjusted to show the effect of modifying weights / biases.

### Loss functions (10 Mins)

Many of our old loss functions are still in play, but sometimes NNs use alternatives like Huber or Hinge.

### Gradient Descent and Batches (5 Mins)

### Optimizers (5 Mins)

Adam will be a good default choice here.

### Tensorflow and Keras (15 Mins)

Much of the syntax will be new here, so it's worth describing the construction of the model step by step.

### Conclusion (5 Mins)

We built our first network! Now we need to be able to
- tweak it
- improve it
- evaluate it
- regularize it
