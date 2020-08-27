---
title: Optical Flow with Michael Black
excerpt: "Resources for computational neuroscience"
categories: [computational neuroscience]
tags: [notes, computational neuroscience, literature review, resources, computer vision]
header:
  teaser: assets/images/posts/starlings2.jpg
toc: true
---

*Notes based on Michael Black's optical flow presentation from 2013*

# What is optical flow?

## Let's get an intuition... 

The term originates from J.J Gibson's *The Ecological Approach to Visual Perception*, in which Gibson calls the motion of luminous patterns across the image sensor an *optic flow*. This pattern of flow gives an organism access to information relevant to survival.

Mathematically, an optic field is a 2D velocity field describing the *apparent* motion of an image. Apparent motion is different from actual motion, which is described by a motion field. Below, we have a Lambertian (matte) ball that reflects light in all directions. Imagine that it's spinning in 3D.

What is the motion field? What is the optical field?

* INSERT IMAGE HERE*

https://www.google.com/url?sa=i&url=http%3A%2F%2Fromain.vergne.free.fr%2Fteaching%2FIS%2FSI04-lighting.html&psig=AOvVaw1tu6M8do-GlELGhQ-6MB7C&ust=1595981466355000&source=images&cd=vfe&ved=0CAIQjRxqFwoTCOjwhbbU7uoCFQAAAAAdAAAAABAJ

The motion field will be a horizontal component. However, the optical flow is zero, because there is no way for the viewer to see that the ball is moving. Thus we can see that motion and optical fields are decoupled from each other.

## Let's get some mathematical precision...

*Enter equations here* 

## Traditional applications

Traditional applications are broad, including pedestrian detection, estimation of camera motion, 3D structure of world, and location of motion boundaries in the scene. Video compression uses optic flow to figure out what's constant in the video and preserves it. You can get the "painterly effect" by computing optical flow over the frames, like in the movie *What Dreams May Come True*, which used optical flow code from Michael Black's PhD thesis.

* INSERT IMAGE HERE * 

*The Matrix* used optical flow for the bullet time sequences, which shot actors from many cameras and computed optical flow between the views of the cameras over time. They were then able to interpolate over any point in space and time fairly realistically. *Matrix Reloaded* also took the code from Black's thesis and applied it to faces, so they were able to produce faces with new expressions.

* INSERT IMAGE HERE * 

# What are the assumptions of optic flow?

Like any model, optic flow is full of caveats.

* **Brightness constancy** - if we look at a small patch of the image, there is a structure that is moving. While the 2D location changes, the pattern itself remains roughly the same. We will make an assumption that the image looks the same, except for a linear offset of the pixels.

* INSERT EQUATION HERE*

* **Spatial smoothness** - Neighboring pixels in an image are likely to belong to a same surface. We will take the four nearest neighbors ofe very pixel and assume they have the same flow. Another way to say this is that the spatial derivative of the optical flow field is zero.

* INSERT EQUATION HERE * 

## What is the formalization?

Objective function:

* Brightness Constancy 

* INSERT EQUATION HERE. * 

We have another assumption here: the quadratic error implies the noise is Gaussian.

Let's define a spatial term, which is a function of the flow field.

* INSERT EQUATION HERE * 

The assumptions here are that the flow field is smooth, deviations from smoothness are Gaussian, first order smoothness is all that matters, and the flow derivative is approximated by first differences. The assumptions here can certainly be improved, but at least give us a basic working model.

### Let's solve our flow...

* INSERT EQUATION HERE * 

Optical flow constraint equation:
Partial derivative of x equation + partial derivative of y function + partial derivative of time = 0

New assumptions
* Flow is small
* Image is differential function
* First order Taylor series is a good approximation

## Aperture Motion Problem

At a single small region of the image, you might not have enough information to disambiguate what the optical flow is. At a single point, you can get an ambiguous constraint.

At each point in an image, we'd like to find a unique point that satisfies all constraints.

## What is the optimization method?

Horn and Schunck '81 in "Determining Optical Flow" derive the above objective function, but could not find a great optimization method to solve their equation. Thus optical flow was deemed ill-posed and unoptimizable for many years. However, people were not looking at all the assumptions...

# Picking off Assumptions

#### Assumption Overview #### 
* Flow is small
* Image is differential function
* First order Taylor series is a good approximation
* Image is differentiable
* Flow field is smooth
* Deviations from smooth are Gaussian
* First order smoothness is all that matters
* Flow derivative is approximated by first differences

In 2013, this was the research frontier: Black proposed on picking off assumptions until the objective function became optimizable. 

## Coarse-to-fine Estimation: Create a pyramid 
Bergen et al. 1992

Assumption: motion is small, first-order Taylor series is a good approximation. 

# How do you evaluate your optic flow method? 
 

### Difficult getting ground truth.

Black had difficulty getting ground truth data with which to evaluate optical flow. A camera won't automatically give you pixel data. Years ago, he went to graphics people in Hollywood for the information to generate their visuals, but the studios that owned the copyright had no interest in giving the data to him. 

Some movies are free, though. The Durian Open Source Movie Project uses Blender and a worldwide community of animators to make movies (e.g. *Sintel*).

## What can animated movies teach us about optical flow in the real world? 
Questions
* Will the results generalize? 
* Is it realistic enough? 

Idea: Lookalikes:
* Real scenes semantically similar to Sintel Scenes
* They looked at marginal image derivative statistics between Sintel and look alikes

Do motions resemble natural motions? 
Compare optical flow on Sintel and lookalikes, compare statistics.

## Modern techniques
* Coarse-to-fine
* Median filtering
* Graduated non-convexity
* Pre-proprocessing
* Bi-cubic interpolation 
* Penalty function

Secrets of optical flow and their principles, 2010.

## Evaluation
* EPE
* AAE

HTe 

# Sources
* [Michael Black's 2013 lecture on optical flow at Max Planck Institute for Intelligent Systems](https://www.youtube.com/watch?v=tIwpDuqJqcE)