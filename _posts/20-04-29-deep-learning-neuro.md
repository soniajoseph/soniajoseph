---
title: "Ideas for the deep learning framework in neuroscience" 
tags: [machine learning, deep learning, computational neuroscience, neuroscience]
excerpt: "On accounting for billions+ of parameters and nonlinearities"
categories: [blog]
header:
  teaser: assets/images/posts/deepdream.png
toc: true
mathjax: "true"
comments: true
collection: posts
---

Broadly speaking, computational neuroscience can be divided into two camps: models that analyze neural data, and models whose structure mathematically imitates that of the brain. The latter fascinates me, acting as a proof-of-concept for various facets of intelligence. And so it is unsuprising that my encounter with neural nets in college sent me away from molecular neuroscience and deep into foundational AI. 

# The Goal-Driven Technique

One technique that falls into this latter camp is the "goal-driven", or "normative" deep learning method. Instead of fitting your deep learning model directly to neural data, you train the model on task that the neural substrate performs. Then, you compare the internal representations between the artificial model with the neural data. If the internal representations correlate throughout the model, then your model may be biologically plausible.

For example, if I wanted to understand the visual stream, I could train a deep CNN to recognize images. Then, I could compare the representations throughout the layers of that artificial neural net to neuron responses in the hierarchy of a macaque visual stream. Because the same problem only has so many solutions, certain artificial neural nets will find a solution that is mathematically analogous to that of the biological visual stream. 

This approach has a major advantage over past methods in that the researcher does not have to manually set the parameters of the model. Rather, the parameters emerge organically after optimizing the model to perform a given task. And perhaps most importantly, not only do the outputs of the model correlate with its biological counterpart; the internal representations of the model also match the biology, without any explicit training on neural data. The model is not a black-box: it is interpretable, in that layers of the CNN [map onto layers of the biological visual stream in terms of predictivity](https://www.pnas.org/content/111/23/8619).

We can examine these task-optimized artificial neural nets further. Perhaps they are similar to the brain in ways that are hard to study in a wet lab, and so they can act as inexpensive models for "virtual" lesion and electrophysiology studies.

Of course, we must also exercise caution-- neural nets and the brain are complex systems, so vastly different internal configurations can lead to the same final outcome. However, with the correct interpretation, this method will lead to rapid advances in our understanding of the brain.

Below I've compiled a list of potential research directions in which to take the goal-driven method.

# Potential Research Directions

## Sparsity

It is well-known that representations in the brain are highly [sparse](https://pubmed.ncbi.nlm.nih.gov/22579264/). The [Lottery Ticket Hypothesis](https://arxiv.org/pdf/1803.03635.pdf), written more about [here](https://soniajoseph.github.io/pruning/), shows that neural nets can retain most of their accuracy depite losing 99% of their weights. Thus fascinating questions arise whether neural pruning mechanisms like [microglia](https://pubmed.ncbi.nlm.nih.gov/21778362/) can be modeled by sparsifying artificial neural nets.

## Movies, Environments, and Naturalistic Vision

Image datasets are limited in that they do not approximate naturalistic vision, which operates in a 3D environment through time. To give rise to maximally biologically-plausible architectures, it is likely that the model will have to be optimized to recognize objects in a 3D environment that it can interact with. Recent research shows that [unsupervised learning on video is a promising approach](https://openaccess.thecvf.com/content_CVPR_2020/papers/Zhuang_Unsupervised_Learning_From_Video_With_Deep_Neural_Embeddings_CVPR_2020_paper.pdf).

More broadly, I suspect that unsupervised methods like [SimCLR](https://arxiv.org/abs/2002.05709) will become default when using the goal-driven method. Unsupervised methods do not need langauage, as organisms do not necessarily need language to learn categories.

## Intermediate "Gabor Patches"

<figure>
  <img src="/assets/images/posts/deepdream.png">
  <figcaption>[Source](https://ai.googleblog.com/2015/06/inceptionism-going-deeper-into-neural.html).</figcaption>
</figure>

Applying gradient descent (["DeepDream"](https://ai.googleblog.com/2015/06/inceptionism-going-deeper-into-neural.html)) on the original image is one method to visualize the stimulus that maximally activates a given neuron of an artificial neural net. Just as Gabor patches drive neural activity in the early layers of the biological visual stream, can visualizations like the one above drive neural activity in intermediate layers of the model's biological counterpart? 

## Ventrolateral PFC in Object Recognition

The role of the PFC in object recognition is poorly understood, but the area is suspected to encode [behaviorally relevant information about the object](https://www.researchgate.net/publication/12122406_Freedman_DJ_Riesenhuber_M_Poggio_T_Miller_EK_Categorical_representation_of_visual_stimuli_in_the_primate_prefrontal_cortex_Science_291_312-316). Some evidence suggests that the ventrolateral prefrontal cortex (vlPFC) is involved in top-down processes that [aid object recognition](https://pubmed.ncbi.nlm.nih.gov/17071109/) and contain [small populations of domain-specific neurons sensitive to eyes, faces, and words](https://psycnet.apa.org/record/1999-03885-004).

One idea is measuring neural response in the vlPFC to an artificial neural net trained on images. We can then manipulate the stimuli to measure activations in the vlPFC. If our goal-driven model is predictive of responses, we can check whether the intermediate layers are also predictive of IT, V4, V2, and V1, in line with the [original findings for the IT](https://www.pnas.org/content/111/23/8619).

*Read more about using goal-driven neural nets on the prefrontal cortex [here](https://soniajoseph.github.io/computational%20neuroscience/deep-learning-models-prefrontal-cortex/)*.

## Mapping Out Architecture Space: CapsuleNets

Which neural net architectures have the highest predictivity of neural response? Is the architecture biologically plausible? 

One interesting candidate is CapsuleNets. [Previous studies](https://www.pnas.org/content/111/23/8619) focus on comparing the internal representation of CNNs to that of the visual stream. However, in failing to encode the spatial relationships between features, CNNs lose biological plausibility. Hinton et al created [CapsuleNets](https://arxiv.org/abs/1710.09829) as one candidate for a more biologically plausible model.

It's possible that the activity vectors and dynamic routing of CapsuleNets provide a more biologically plausible mechanism than the normalization, pooling, thresholding, and convolutions of CNNs. One promising avenue is using the goal-driven method to compare the the similarity between task-optimized CapsuleNets and CNNs to biological neural responses. If CapsuleNets are more predictive of neural response, that may speak favorably for their biological plausibility.

## Performance and Predictiveness: the Upside-Down U-shape

Artificial neural nets do not necessarily face the same biological constraints as brains. One example may be an architecture like ResNets, which improves image classification performance but may not be biologically plausible with its long skip-connections.

Thus it is not surprising that the top-performing neural nets on image classification [surpass human-level performance](https://paperswithcode.com/sota/image-classification-on-imagenet). One hypothesis is that the relationship between performance on image classification and predictivity of neural activity follows an upside-down U-shape. Neural nets that are better at image classification will correlate more strongly with IT response-- until an inflection point, when the neural net will surpass biological limitations. From thereon I expect that the correlation to IT response will decrease. 

## Integration with Energy-Based Models

Long used as [models for associative memory](https://bi.snu.ac.kr/Courses/g-ai09-2/hopfield82.pdf), energy-based models like Boltzmann machines and Hopfield networks are [slowly making a comeback in modern deep learning](https://arxiv.org/abs/2008.02217). It would be interesting to examine hybrids between deep learning models and energy-based models as biologically plausible candidates.

# More Resources

Some good overviews of the goal-driven paradigm are [this original 2014 paper](https://www.nature.com/articles/nn.4244), and [this 2019 overview](https://oxfordre.com/neuroscience/view/10.1093/acrefore/9780190264086.001.0001/acrefore-9780190264086-e-46).

The technique is flourishing cross computational neuroscience labs across the world, so I will be missing many labs if I merely listed the ones I knew.

If you're looking for a more detailed, hands-on introduction to goal-driven neural nets, Carsen Stringer's [Jupyter notebook](https://github.com/NeuromatchAcademy/course-content/blob/master/tutorials/W3D4_DeepLearning1/W3D4_Tutorial3.ipynb) under Neuromatch Academy is an unrivaled resource. Other notebooks in the series also include detailed tutorials on the basics of machine learning and PyTorch.

Please message me if you would like to chat or collaborate in the future on any of these ideas (or none!). 

*Note: Citations updated on July 2020.* 
