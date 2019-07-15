---
title: "Facial Recognition with Collaborative Representation-Based Classification"
excerpt: Facial recognition with collaborative representation-based classification.
categories: [machine learning]
tags: [machine learning, artificial intelligence, image classification]
header:
  teaser: assets/images/posts/first_face_fifteen_subjects_teaser.png
toc: true
---

*The following post assumes some prior background in machine learning, such as familiarity with ordinary least squares (OLS).*

## Let's talk about faces...

Collaborative representation-based classification is a neat way to implement ordinary least squares regression (which has continuous output) in order to classify new data points discretely. I am writing about this algorithm because it is elegant, but undertaught.

We'll use the YALE face database, consisting of 15 subjects in 11 slightly different poses, for a total of 165 64x64 pixel images, which I downloaded [here](http://www.cad.zju.edu.cn/home/dengcai/Data/FaceData.html) [^1][^2][^3][^4]. 
<figure class="half">
    <a href="/assets/images/image-filename-2-large.jpg"><img src="/assets/images/posts/first_face_fifteen_subjects.png"></a>
    <a href="/assets/images/image-filename-1-large.jpg"><img src="/assets/images/posts/first_11_faces_of_one_subject.png"></a>
</figure>
The data points $${x_0... x_n}$$ are arranged in the columns of matrix $$X$$ (size 4096x165) so that the first 11 columns correspond to the 11 images for the first face, the next 11 columns to the second face, and so on.

## Theory

We begin by implementing ordinary least squares regression on $$X$$, in which we find $$w$$ to minimize the residual:

$$ \begin{equation}
    w* = 
\arg\min_{w} \lVert \mathbf{Xw - x} \rVert_2^2

\end{equation} $$

We solve the normal equation to get $$\begin{equation} w* \end{equation}$$:

$$ \begin{equation}
    w* = (X^TX)^{-1}X^Tx

\end{equation} $$

The first 11 elements of $$w$$, which we will call $$w_0$$, represent how much of each column of the first face we need to represent a new data point $$x$$; the next 11 elements $$w_1$$ represent how much of each column of the second face we need to represent $$x$$; and so on.

Likewise, $$w_0$$ corresponds to the first 11 columns of $$X$$, which we will call $$X_0$$. 

To predict the face for each data point $$x$$, we take the dot product $$\hat{y_i}$$ of $$X_i$$ and $$w_i$$ for each face $$i$$. We calculate the Euclidean distance between $$x$$ and $$\hat{y_i}$$. The face $$j$$ out of $$i$$ that yields the lowest distance will determine the label.

$$ \begin{equation}
    j = \arg\min_{} \lVert \mathbf{x - \hat{y_i}} \rVert_2
\end{equation} $$

## Code

The code for the algorithm is as follows:

```python
def collab_rep(X_train, X_test, Y_train, Y_test, lambda_weight=0):
    
    num_test_samples = len(Y_test)
    num_classifications = 15
    
    # least squares
    X = np.matrix(X_train)
    w = np.linalg.pinv(np.dot(X.T,X)) * (X.T) 
    
    predicted_label = np.zeros((num_test_samples,))
    
    # for each new vector
    for i in range(0,num_test_samples):
        test_ex = np.matrix(X_test[:,i]).T
        p = w * test_ex
        
        # compare the new vector to linear combinations of each face j
        dist = np.zeros((num_classifications,))
        for j in range(0,num_classifications):
            m = X[:,j*num_classifications:(j+1)*num_classifications]
            w_j = p[j*num_classifications:(j+1)*num_classifications]
            reconstructed = np.dot(m, w_j)
            dist[j] = np.linalg.norm(reconstructed - test_ex)
        
        # classify new vector according to minimum distance between the vector and reconstructed 
        predicted_label[i] = np.argmin(dist)
        
    # calculate the accuracy
    test_err = np.count_nonzero(predicted_label - Y_test)
    test_acc = 1-(test_err/100.0)
    
    return test_acc
```

We can run the code with 50-fold cross-validation to get a **55%** accuracy, far better than chance (>7%). 

```python
# load YALE database from http://www.cad.zju.edu.cn/home/dengcai/Data/FaceData.html
from sklearn import preprocessing

# load data and transpose so that columns are faces
YALE = io.loadmat('Yale_64x64.mat')
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
    accuracy = collab_rep(fea_Train, fea_Test, gnd_Train, gnd_Test, lambda_weight = 0)
    accuracy_list.append(accuracy)

print(accuracy_list)
print("accuracy: ", np.mean(accuracy_list))
```

## Conclusion

In sum, the vectors of each face category "collaborate" via linear combinations in order to classify new faces (i.e., face category with the closest collaboration to the new face "wins"). We can further modify the algorithm to incorporate sparsity or lasso.

Collaborative representation-based classification is too inefficient to use practically compared to neural nets. Nonetheless, it is an elegant discrete implementation of a continuous algorithm.

*Full code on Github [here](https://github.com/soniajoseph/Collaborative-Representation-Based-Classification).*

## References
[^1]: Deng Cai, Xiaofei He, Yuxiao Hu, Jiawei Han and Thomas Huang, "Learning a Spatially Smooth Subspace for Face Recognition", CVPR'07. Bibtex source
[^2]: Deng Cai, Xiaofei He and Jiawei Han, "Spectral Regression for Efficient Regularized Subspace Learning", ICCV'07.	Bibtex source
[^3]: Deng Cai, Xiaofei He, Jiawei Han, and Hong-Jiang Zhang, "Orthogonal Laplacianfaces for Face Recognition", IEEE TIP 2006. Bibtex source
[^4]: Xiaofei He, Shuicheng Yan, Yuxiao Hu, Partha Niyogi, and Hong-Jiang Zhang, "Face Recognition Using Laplacianfaces", IEEE TPAMI 2005. Bibtex source