# Torch-Dreams 
> A python package to reverse engineer neural nets for interpretability. 

## Summary
When a deep-learning model looks for an eye, what does it look for ? Does it look for eyebrows ? or eyeballs maybe ? This is exactly what torch-dreams can help us answer.

<p align="center">
<img src = "https://github.com/Mayukhdeb/torch-dreams/blob/joss-paper/images/paper/face_segmentation_breakdown_with_torch_dreams.jpg?raw=true"> Figure 1: A breakdown of a facial segmentation model’s classes with torch-dreams
</p>

Torch-Dreams is a tool that helps visualize what each layer/channel/unit within a neural-net looks for in an input image. These visualizations help in breaking the black-box like nature of neural nets and helps gain important insights on how the model actually works. 

## Statement of need

Comparing different neural net architectures has been mostly based on their performance metrics like accuracy, loss and response times. The common element in all of these approaches is that they treat the different models as “black-boxes”. This approach tells us about “how good a model is” but not “how a model thinks”. 

Torch-dreams helps us gain an insight on how a model thinks. This is done by optimizing the input image to maximize the activations of user selected elements(layers/channels/units) within the neural net. It relies heavily on packages like PyTorch, NumPy[1] and openCV. 

The core inspiration behind this library has been the DeepDream algorithm[2], but torch-dreams scaled it up to give the user much more freedom in terms of model architectures and customizability. 

## Usage


Torch-dreams can be used to perform gradient ascent on input images to maximize the activations of certain layers/channels/single units within the neural net. In order to reduce high frequency patterns, it uses various methods such as scaling, random jitter, random rotations, etc. All of which can be adjusted by the user as per required.

One of the key advantages of this library is the amount of freedom that it provides. The user can write their own custom optimization functions to optimize the input image in a virtually infinite number of ways. 

One of the examples could be “channel algebra” where we simultaneously optimize the sum or the difference of multiple channels in a single input image.  

<p align="center">
<img src = "https://github.com/Mayukhdeb/torch-dreams/blob/joss-paper/images/paper/algebra_1.jpg?raw=true">
</p>

<p align="center">
<img src = "https://github.com/Mayukhdeb/torch-dreams/blob/joss-paper/images/paper/algebra_2.jpg?raw=true"> Figure 2:Optimizing the sums and differences of channel activations
</p>

The user can also make and use custom gradient masks with numpy to spatially blend or restrict certain optimization functions.

<p align="center">
<img src = "https://github.com/Mayukhdeb/torch-dreams/blob/joss-paper/images/paper/grad_mask.jpg?raw=true">
Figure 3: Optimizing 5 different channels with gradient masks
</p>

Torch-dreams is simple enough for artists to generate fascinating patterns, at the same time it’s also flexible enough for researchers to run large experiments.