---
title: "Facial Recognition with Collaborative Representation-based Classification"
excerpt: Facial recognition with collaborative representation-based classification.
categories: [machine learning]
tags: [machine learning, artificial intelligence, image classification]
toc: true
---
## Let's talk about faces...

Collaborative representation-based classification is a neat way to use least squares regression (which has continuous output) in order to classify new data points discretely. This algorithm is elegant, but undertaught.

We'll use the YALE face database, consisting of 15 subjects in 11 slightly different poses, for a total of 165 images, which I downloaded [here](http://www.cad.zju.edu.cn/home/dengcai/Data/FaceData.html) [^1][^2][^3][^4]. 

<figure class="half">
    <a href="/assets/images/image-filename-2-large.jpg"><img src="/assets/images/posts/first_face_fifteen_subjects.png"></a>
    <a href="/assets/images/image-filename-1-large.jpg"><img src="/assets/images/posts/first_11_faces_of_one_subject.png"></a>
</figure>

The data points are arranged in columns of matrix *X* so that the first 11 columns correspond to the 11 images for the first face, the next 11 columns to the second face, and so on. 

We first implement an ordinary least squares regression, in which we select *w* to minimize the residual:

EQUATION HERE

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