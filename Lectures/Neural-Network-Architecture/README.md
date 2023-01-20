# Neural Network Architecture

Overview of network architecture including activation functions.

## Learning Goals

- Describe the basic structure of densely connected neural networks
- Describe the various activation functions that are used in neural networks

## Lesson Materials

Two notebooks:
- [Jupyter Notebook: Neural Network Architecture](neural_network_architecture.ipynb)
- [Jupyter Notebook: Neural Network from Scratch](neural_network_from_scratch.ipynb)

**Note from Greg:** the NN from Scratch notebook is in both this lecture and [the next on intro to tensorflow](../IntroKerasTensorflow). Current plan is to go through the first part for this session and the second part for the next (since it's not until the second that we touch on backpropagation). I didn't want to get rid of the sample training that we do with the digits data since that figures into the content of the second and third lectures on neural networks.

## Lesson Plan

### Introduction (10 Mins)

Historical background and relation to previous models, esp. incl. logistic regression and model stacking.

### Basic Architecture (20 Mins)

- layers
- nodes
- dense connections
- biological analogy

### Forward Propagation (25 Mins)

Mechanics of linear transformation + bias + activation function. Compare to generalized linear model (esp. logistic regression).

Variety of activation functions. There's a little material in the notebook about when one might want to use each.

### Conclusion (5 Mins)

Next up: calculating gradient of loss function and then updating weights!
