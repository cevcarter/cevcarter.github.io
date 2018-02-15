---
layout: post
title: "Variational Inference for Neural Networks"
---

<iframe src="https://drive.google.com/drive/u/1/my-drive"></iframe>


Variational Inference is a technique based on Kullback-Liebler divergence. The technique seeks to arbitrarily approximate a given target distribution with another simpler distribution. this makes sampling from the distribution in question much simpler to do, which is helpful when sampling from the complex posterior distributions that are often present when making inferences about neural networks. 

To explore this concept, I built a simple neural network in Theano to classify objects in a simple dataset. I then used PyMC3 to approximate the posterior distribution of the weights of the neural network and subsequently sample from the posterior for the sake of inference. 

This gave an interesting sense of understanding about the network itself; ordinarily networks are treated like black boxes, but being able to visualize the posterior over the weights of the network was a bit like "seeing" the network itself. This kind of information allows the algorithm designer to quickly assess uncertainty in the weights if he or she wanted to, or build up a more formal description of the uncertainty in the network if such an analysis was deemed necessary. 

Since this is an ongoing project, I won't have any code up for a while, but I hope to have some interesting results up soon.  

