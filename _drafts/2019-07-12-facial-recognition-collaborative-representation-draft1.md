---
title: "Machine Learning for All Audiences: Facial Recognition with Collaborative Representation"
excerpt: Facial recognition with collaborative representation-based classification for all audiences.
categories: [machine learning]
tags: [machine learning, artificial intelligence, image classification, all audiences]
toc: true
---

The fastest way I understand machine learning is visually. I often find that a clever diagram or visualization will unlock my understand far more quickly than staring at equations. In this post, I try to explain *collaborative representation-based classification* as clearly and as visually as possible.

While this post should be accessible to anyone without background in programming or higher-level math, I have also included code for the nitty-gritty Python implementation. The code might not be necessary for your purposes: Feel free to skip it if you just want the high-level intuition.


## Let's talk about faces...

Collaborative representation-based classification is a neat way to use linear regression (an algorithm with continuous output) in order to classify new data points discretely. I like collaborative representation-based classification because of its visual elegance and [Tetris effect](https://en.wikipedia.org/wiki/Tetris_effect) on my thinking.

We'll use the YALE face database, consisting of 15 subjects in 11 slightly different poses, for a total of 165 images, which I downloaded [here](http://www.cad.zju.edu.cn/home/dengcai/Data/FaceData.html) [^1][^2][^3][^4]. 

<figure class="half">
    <a href="/assets/images/image-filename-2-large.jpg"><img src="/assets/images/posts/first_face_fifteen_subjects.png"></a>
    <a href="/assets/images/image-filename-1-large.jpg"><img src="/assets/images/posts/first_11_faces_of_one_subject.png"></a>
</figure>

Let's take some faces and plot them on three axes:

INSERT FACE IMAGES HERE


What do these three axes mean? Each axis represents the value of a pixel from 0 (black) to 255 (white). Taken together, the axes represent all possible black-and-white images that consist of just three pixels. 

"But face images are way more than just three pixels," you might say. This is true-- the images we are using are precisely 1024 pixels (32x32). So instead of plotting the images on a three-dimensional space, we should plot them on a 1024-dimensional space. However, 1024-dimensional space defies our human imaginations: We cannot visualize it. Therefore, for this tutorial, we shall represent this space with our simple three axes. Just keep in mind that this space must actually be 1024 dimensions in order to account for the total number of pixels for each image.

Each image has a corresponding vector. For example, this guy: 

PHOTO

can be represented by [1.0, 0.5, 0.7]. Again, remember that the vector actually has 1024 entries, called *components*, in order to represent its 1024 pixels, but we are pretending that there are three components so that we can understand later concepts.

Now say we are using this algorithm on social media, and this guy uploads a new photo of himself:

PHOTO OF GUY

The algorithm has to identify this guy's face out of the library of all his friends faces (which includes the guy's own face) so that it can label the face with the correct name. (I should note that the real  Facebook uses neural nets, which are far more accurate and sophisticated than collaborative representation, which is way too inefficient. Nonetheless, collaborative representation will work with reasonable accuracy.)

We turn our new

So what does our social media site do? We turn the new face into a vector, and compare the vnew ector to the existing face vectors, one at a time. We find the existing face vector closest to the new face vector, and then we will give the label of the close to the new face vector.

We do this by taking *linear combinations* of the existing face vectors

In the social world, you can use linear combinations of past people to represent new people (note I do not think this is the best way to understand people; it is not the most integrated and holistic, but I found that this was what my mind was doing). Another way to think about this linear combination of past people is a weighted average of past people.

Test
```python

# load YALE database from http://www.cad.zju.edu.cn/home/dengcai/Data/FaceData.html
from sklearn import preprocessing

# load data and transpose so that columns are faces
YALE = io.loadmat('Yale_32x32.mat') 
fea = YALE['fea'].T
gnd = YALE['gnd'].T

# scale pixels to be [0,1]
maxValue = np.amax(fea)
fea = fea / maxValue

accuracy_list = []
for i in range(1,51,1):
    subset = io.loadmat('8Train/' + str(i) + '.mat') 
    trainIdx = subset['trainIdx'].flatten()
    testIdx = subset['testIdx'].flatten()
    
    # generate training and test data
    fea_Train = fea[:,trainIdx-1]; 
    fea_Test = fea[:,testIdx-1]; 

    gnd_Train = gnd[:,trainIdx-1]; 
    gnd_Test = gnd[:,testIdx-1]; 
    
    # get accuracy via collaborative representation
    accuracy = collab_rep(fea_Train, fea_Test, gnd_Train, gnd_Test, lambda_weight = 100000)
    accuracy_list.append(accuracy)

print(accuracy_list)
print("accuracy: ", np.mean(accuracy_list))

```


The equation form of the above is:

## Sparse Approximation 

*We don't want these faces because they're convicts, or something.*

One of my favorite machine learning algorithms is sparse approximation, perhaps because it is so simple. As with most algorithms I've come to deeply understand, it began patterning my thoughts in other domains, like the social world.

The idea is simple. We have a library of faces, as below:

Our library contains ___ faces.

## Sparse Representation

## References
[^1]: Deng Cai, Xiaofei He, Yuxiao Hu, Jiawei Han and Thomas Huang, "Learning a Spatially Smooth Subspace for Face Recognition", CVPR'07. Bibtex source
[^2]: Deng Cai, Xiaofei He and Jiawei Han, "Spectral Regression for Efficient Regularized Subspace Learning", ICCV'07.	Bibtex source
[^3]: Deng Cai, Xiaofei He, Jiawei Han, and Hong-Jiang Zhang, "Orthogonal Laplacianfaces for Face Recognition", IEEE TIP 2006. Bibtex source
[^4]: Xiaofei He, Shuicheng Yan, Yuxiao Hu, Partha Niyogi, and Hong-Jiang Zhang, "Face Recognition Using Laplacianfaces", IEEE TPAMI 2005. Bibtex source