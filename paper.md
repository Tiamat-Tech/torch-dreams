# Torch-Dreams 
> A python package to reverse engineer neural nets for interpretability. 

## Summary
When a deep-learning model looks for an eye in an image, what does it actually look for? Does it look for eyebrows? or eyeballs maybe? This is exactly what `torch-dreams` can help us answer.

![ A breakdown of a facial segmentation model’s classes with `torch-dreams`](https://github.com/Mayukhdeb/torch-dreams/blob/joss-paper/images/paper/face_segmentation_breakdown_with_torch_dreams.jpg?raw=true)

`torch-dreams` is a tool that helps visualize what each layer/channel/unit within a CNN looks for in an input image. It aims to make model interpretability more accessible and open for general use. 


## Statement of need

With the advent of CNNs into fields like healthcare and developmental biology, it has been more important than ever to understand and visualize the representations learned by the models to justify the decisions taken by it. 

Different model architectures are generally compared based on performance metrics like accuracy, loss and speed. The common element in these approaches is that they tell us “how good a model is” but not “how a model thinks”. 

`torch-dreams` helps us gain an insight on how a model thinks. This is done by optimizing the input image to maximize the activations of user selected elements(layers/channels/units) within the neural net. It relies heavily on packages like PyTorch, NumPy[1] and openCV. 

The aim of this library was inspired by the optvis module in tensorflow/lucid[3] and the the Feature Visualization[2] paper, but torch-dreams has been re-written from scratch to give the user complete freedom over selection of layers/channels/units and determining how exactly should the activations be optimized. 

## Usage

`torch-dreams` can be used to perform gradient ascent on input images to maximize the activations of certain layers/channels/single units within the neural net. In order to reduce high frequency patterns, it uses various methods such as scaling, random jitter, random rotations, etc. All of which can be adjusted by the user as per required.

One of the key advantages of this library is the amount of freedom that it provides. The user can write their own custom optimization functions to optimize the input image in a virtually infinite number of ways. 

A good example would be “channel algebra” where we simultaneously optimize the sum or the difference of multiple channels in a single input image.   

![](https://github.com/Mayukhdeb/torch-dreams/blob/joss-paper/images/paper/algebra_1.jpg?raw=true)

![Optimizing the sums and differences of channel activations](https://github.com/Mayukhdeb/torch-dreams/blob/joss-paper/images/paper/algebra_2.jpg?raw=true)

The user can also make and use custom gradient masks with numpy to spatially blend or restrict certain optimization functions.

![Optimizing 5 different channels with gradient masks](https://github.com/Mayukhdeb/torch-dreams/blob/joss-paper/images/paper/grad_mask.jpg?raw=true)

`torch-dreams` is simple enough for artists to generate fascinating patterns, at the same time it’s also flexible enough for researchers to run large experiments.

## Acknowledgements

We acknowledge the support and feedback from team amFOSS and Gene Kogan during the genesis of this project. 