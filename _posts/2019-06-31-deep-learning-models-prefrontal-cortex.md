---
title: "Deep Learning Models for the Prefontal Cortex in Object Recognition"
excerpt: A hypothetical proposal for using the goal-driven neural network paradigm by Yamins and DiCarlo to model object categorization in the ventral stream.
categories: [computational neuroscience]
tags: [notes, computational neuroscience, neuroscience, neural dynamics, deep learning, resources]
header:
teaser: assets/images/posts/starlings2.jpg
toc: true
classes: wide
---

*A hypothetical proposal written in Spring 2018 for the neuroscience concentration, advised by Dr. Jonathan Pillow.*

## Abstract
The ventral stream underlies object recognition and categorization in human and non-human primates, but neural encoding in its highest region, the prefrontal cortex (PFC), remains poorly understood. With recent advances in deep learning, goal-optimized convolutional neural networks (CNNs) have been shown to be highly accurate for predicting neural responses in the highest ventral cortical area, the inferior temporal (IT) cortex, which has high category selectivity based off visual information. Furthermore, the model’s intermediate layers were highly accurate for predicting upstream ventral neural responses (in V4). We aim to apply goal-driven CNNs to predicting neural responses in the ventrolateral prefrontal cortex (vlPFC). Because research suggests that the vPFC incorporates behaviorally-related object information in object recognition, we hypothesize that goal-driven CNNs will underperform data-driven CNNs on vlPFC neural responses. We hope that our results will shed light on the role of the vlPFC in the ventral stream and the extent to which the region encodes for behaviorally-related object features.

## Background
### Object Categorization and the Ventral Stream
An important area in computational neuroscience is developing models for object recognition and categorization--  the ability to recognize and categorize an object despite variation in orientation, size, illumination, and other sources of noise (Dicarlo, Zoccolan, & Rust, 2012). The ventral stream (the “what” pathway) supports object recognition and categorization in humans, with homologous areas in non-human primates (Mishkin, Ungerleider, & Macko, 1983; Malach, Levy, & Hasson, 2002; Kriegeskorte et al., 2008). Visual information travels from the retina, to the lateral geniculate nucleus of the thalamus (LGN), to V1, to V2, to V4, to the inferior temporal cortex (IT), and finally to the prefrontal cortex (PFC), which links object perception to memory and action (Felleman & Van Essen, 1991; Serre, Oliva, & Poggio, 2007). Past research suggests that the ventral stream operates hierarchically, encoding object identity and category more explicitly with each successive area (DiCarlo, Zoccolan, & Rust, 2012). More specifically, object position and scale (Riesenhuber & Poggio, 1999), invariance (Ullman, 1997), size of receptive field (Kobatake & Tanaka, 1994), and complexity of optimal stimuli for the neuron (Kobatake & Tanaka, 1994) increase downstream from V1 to IT. 

Two empirical observations about the ventral stream are relevant: first, it is composed of a series of anatomically distinguishable, connected areas, and second, initial neural activity cascades along these areas (Malach, Levy, & Hasson, 2002). In each stage of the cascade, simple neural operations are performed, including a weighted linear sum of inputs and nonlinearities like activation thresholds and competitive normalizations (Carandini, 2005). When the stages are applied in series, complex nonlinear transformations can arise (Sharpee, Kouh, & Reynolds, 2013). 

### Using Goal-Driven CNNs to Understand the Ventral Stream
Neural encoding models predict spike train responses given novel stimuli (Paninski, Pillow, & Lewi, 2007). It has been proposed that conceptually compelling neural encoding models should have the following three characteristics: stimulus-computability, accepting random stimuli within area of interest; mappability, plausibly corresponding to a biological system; and predictivity, predicting stimulus responses for a randomly chosen neuron in the mapped area (Yamins & Dicarlo, 2016). Given these three requirements, hierarchical convolutional neural networks (CNNs) lend themselves to be suitable models for neural encoding in the ventral stream. CNNs consist of many individual units stacked in layers and perform neuronally plausible operations, such as forming 2D convolutions over input, applying pooling, and adding non-linearities to upstream responses (Yamins & DiCarlo, 2016). In meeting these characteristics, the CNN must have layers that map onto the biological system of interest. For example, a CNN modeling IT response during object recognition must also have intermediate layers that map onto areas of the ventral stream upstream of the IT (e.g. V1, V4), and accurately predict responses in these areas. 

Yamins and DiCarlo (2016) suggest that goal-driven CNNs, which are based off transfer learning and optimized for a behavioral task instead of for neural data, constrain the hidden layers of the network to behave like neurons in preceding areas. Indeed, in CNNs that achieve near-human-level performance on object categorization, the top hidden layers were shown to be the most quantitatively accurate model for neural encoding in the IT without being explicitly trained on neural data (Yamins et al., 2014). Furthermore, the same model’s intermediate layers were state-of-the-art predictors for V4 responses, outperforming previous models (Yau, Pasupathy, Brincat, & Connor, 2012), and its lower layers were predictors for V1-V3 voxel data (Khaligh-Razavi & Kriegeskorte, 2014; Guclu & Gerven, 2015) In sum, the CNN’s top-down, goal-based constraint of object categorization modeled the visual stream from V1 to IT more effectively than past models, without any training on neural data.

### Using Goal-Driven CNNs to Understand Role of PFC in Ventral Stream
Neither goal-driven nor data-driven CNNs have been used to model neural response in the PFC during object categorization. The role of the PFC in object categorization is little understood (Freedman, Riesenhuber, Poggio, & Miller, 2003), but the area is suspected to encode behaviorally relevant information about an object, such as attributes related to response-selection, more strongly than the IT (Freedman, Riesenhuber, Poggio, & Miller, 2003). Other research suggests that the ventrolateral prefrontal cortex (vlPFC) is involved in top-down processes aiding object categorization, including long-term memory retrieval, working memory, and covert attention (Ganis, Schendan, & Kosslyn, 2007), and contains small populations of domain-specific neurons sensitive to eyes, faces, and words (Scalaidhe, 1999). Levy & Wagner (2011) found that the right vlPFC is heterogeneous in function, but the area is largely not understood.

In this paper, we propose extending the work of Yamins et al. (2014) to assess biologically plausible, human-level performance, goal-driven CNNs against measured neural response data in the vlPFC of a non-human primate. We wonder if there is a strong correlation between object categorization task performance and prediction of vlPFC neural responses, similar to the correlation between performance and prediction of IT neural responses. If our goal-driven CNN model is predictive of vlPFC neural responses, we wonder if its intermediate layers are also predictive of IT, V4, V2, and V1, in line with the work of Yamins et al. (2016).

We see reason why a goal-driven CNN may not be as predictive of vlPFC neural responses as of IT neural responses during object categorization. As past research suggests, the vlPFC may be more responsible than the IT for encoding behaviorally-related information from working memory, long-term memory, and attention. For example, in recognizing and categorizing a tool, a primate may not use just visual information, but also contextual information, such as past memories of a tool or a tool’s ability to be picked up. Therefore, while a goal-driven CNN may have near-human-level performance for object categorization, we hypothesize that the model will predict neural responses in the vlPFC less accurately than data-driven CNNs for the same area, and less accurately than goal-driven CNNs for the IT and upstream ventral areas.

## Specific Aims 
We hypothesize that because the vlPFC shows functional heterogeneity and uses contextual information beyond visual features to categorize objects, a goal-driven CNN will underperform a data-driven CNN in predicting vlPFC neural responses. We further suspect that the goal-driven CNN will predict vlPFC neural responses less accurately than IT neural responses. We hope that the results of our tests will shed light on category selectivity in the vlPFC and the role of the vlPFC in object categorization.

We aim to record vlPFC and IT neurons in anesthetized macaque monkeys in response to natural images using multiple electrode arrays. Then, we aim to correlate the performance of CNN models trained on the same images with their vlPFC neural predictivity. We will obtain neural reference points on categorization performance by training linear classifiers on vlPFC and IT neural activity. We will use hierarchical modular optimization (HMO) to create a goal-driven CNN model and measure its vlPFC predictivity at various layers. Lastly, to understand vlPFC and IT activity at the population level, we use a representation dissimilarity matrix (RDM). 

## Description of the Proposed Research 
### Data Collection
We propose collecting neural data from the IT and vlPFC of two awake, behaving rhesus macaques using parallel multielectrode array electrophysiology recording systems under appropriate animal care guidelines in a procedure adapted from Yamins et al. (2014) and Safavi et al. (2018). Around 100 neurons in each area will be selected as visually driven with a separate image set. Animals will be presented with an image set that exposes key characteristics of visual representations with levels of variation in object scale, pose, and illumination (Tiny ImageNet, 2017). We will contort the images for three variation levels using parameters chosen uniformly at random. Objects will have low variance, with constant scale and fixed pose; medium variance, 30% bigger or smaller and rotated less than 45 degrees; and high variance, 60% bigger and smaller rotated less than 90 degrees.

Images will be presented in pseudorandom order for 100 ms each, a duration comparable to natural fixation (DiCarlo & Maunsell, 2000). Images will be presented one at time on a screen followed by a 100 ms blank period. A video tracking system will monitor the macaque’s eye movements, and a juice reward will be given when fixation is maintained for six successive image fixations. Firing rates will be obtained for each image and electrode by averaging spike counts 70-170 ms after stimulus presentation (Majaj, Hong, Solomon, & DiCarlo, 2012). Background firing rate, which is the mean firing rate during blank intervals, will be subtracted from this calculation. Neuron output responses will be obtained from each site by averaging over image repetitions.

### Correlation between Object Recognition and vlPFC Neural Predictivity
We will draw several thousand randomly sampled models from parameter space N. For each model, we will compute linear classifiers for performance and linear regressors for vlPFC and IT predictivity. Using a metaparameter optimization algorithm like Hyperopt (Wistuba, Schilling, & Schmidt-Thieme, 2015), we will optimize models for goal-performance, vlPFC performance, and IT performance and examine the correlation explained for vlPFC and IT neural response variation. We expect that the correlation between goal-performance and vlPFC predictivity will be lower than between goal-performance and IT predictivity, which could suggest that goal-driven models are less accurate for the vlPFC due to the area having greater heterogeneity and access to object behavioral attributes. We also expect that the best-performing data-optimized models will predict vlPFC neural output equivalently or better than goal-optimized models, while the reverse does not hold. This would be the inverse result of the IT neural response prediction accuracy found by Yamins et al. (2014).

The result should shed light on category selectivity in the vlPFC, which so far remains unclear. If, counter to our hypothesis, goal-optimized models perform the same as or better than data-optimized models, but the reverse does not hold, our result could suggest that the vlPFC also exhibits category selectivity similar to the IT.

### vlPFC as Neural Performance Target
We will obtain neural benchmarks on categorization performance at different levels of object variation so that we can understand a key axis of recognition difficulty. We will train linear classifiers on vlPFC and IT data for low, medium, and high variation images. Linear regressor results will be computed with a 10-fold cross-validation with 50/50 training/test splits. A partial least squares (PLS) regression procedure will narrow down the number of weights. In controlling for lower-level confounds, we will also fit other computational models of the ventral stream, including SIFT (a baseline in computer version) (Lowe, 2004), V1 (Pinto, Cox, & Dicarlo, 2008), and HMAX (Mutch & Lowe, 2008). We expect that the performance of all models will decrease as variation increases, while the performance of the IT and vlPFC models will decrease less rapidly, given that they model downstream visual processing, which has high variation tolerance.

### Measure vlPFC Predictivity for Layers of High-Performance Goal-Driven CNN (HMO Model)
We will use hierarchical modular optimization (HMO) to obtain a high-performing goal-driven CNN (Yamins et al., 2014). The HMO is analogous to an adaptive boosting procedure interwoven with hyperparameter optimization. With the same classifier test criteria described in the preceding section, we will train and test the HMO model on the image set. We expect HMO model performance to perform well given large amounts of variation, which is comparable to human-level object categorization.
Then, we will measure the predictivity of layers in the HMO for vlPFC and IT neural responses. We expect that each successive layer of the HMO model will predict neural responses in both regions increasingly well. However, we expect that prediction for IT will be more accurate than vlPFC, because we hypothesize that vlPFC neural responses include behaviorally-related features not available to the goal-driven HMO during training. Even if vlPFC prediction accuracy is worse than IT neural prediction accuracy, we will still see whether the top-performing layer of the HMO for the vlPFC will be the most accurate neural encoding model for vlPFC response during object categorization to date.

### Calculate vlPFC Population Representation Similarity 
We will characterize vlPFC and IT neural activity at the population level. The result may provide insight into vlPFC categorization selectivity, which is currently unclear. We will use a representation dissimilarity matrix (RDM), which can compare two representations of a stimulus in a task-independent manner. Given neuron response vectors R = r1,...,  rkwhere rijis the response of the jth neuron to the ith stimulus, the RDM is defined as RDM(R)ij =1-  cov(ri, rj) var(ri) var(rj)  

The RDM, taken over all stimuli, characterizes the layout of images in the high-dimensional neural population space. In line with the results of Yamins et al. (2014), we expect that the RDM for the IT neural population will have a clear block-diagonal and off-diagonal structure when images are categorically ordered. This structure is associated with IT’s high category selectivity. We are unsure about RDM structure regarding the vlPFC neural population, but we suspect that the matrix will display less block-diagonal and off-diagonal structure than that of the IT due to less category selectivity in the vlPFC based off solely visual information. 

## Methods in Sum
Overall, we expect our results will suggest that the vlPFC incorporates behaviorally-related object features that cannot be derived from visual information. More specifically, we expect that data-optimized CNN models will outperform goal-optimized CNN models for vlPFC neural response during object recognition, because the vlPFC response reflects features that the goal-optimized CNN model does not capture. In line with previous work on hierarchical models of the ventral stream, we expect predictivity for vlPFC to increase with each layer of the goal-optimized CNN. However, we expect the layers of the goal-optimized CNN to better capture neural response in the IT than in the vlPFC due to the former’s high category selectivity based off visual information. We hope that our results will shed light on the role of the vlPFC in object categorization, and the extent that it encodes for behavioral features, which has previously been unclear.

## References
Carandini, M. (2005). Do We Know What the Early Visual System Does? Journal of Neuroscience,25(46), 10577-10597. doi:10.1523/jneurosci.3726-05.2005

DiCarlo, J., & Maunsell, J. (2000). Inferotemporal representations underlying object recognition in the free viewing monkey. Society for Neuroscience.

Dicarlo, J., Zoccolan, D., & Rust, N. (2012). How Does the Brain Solve Visual Object Recognition? Neuron,73(3), 415-434. doi:10.1016/j.neuron.2012.01.010

Felleman, D. J., & Essen, D. C. (1991). Distributed Hierarchical Processing in the Primate Cerebral Cortex. Cerebral Cortex,1(1), 1-47. doi:10.1093/cercor/1.1.1

Freedman, D. J., Riesenhuber, M., Poggio, T., & Miller, E. K. (2003, June 15). A Comparison of Primate Prefrontal and Inferior Temporal Cortices during Visual Categorization. Retrieved from http://www.jneurosci.org/content/23/12/5235.long

Ganis, G., Schendan, H. E., & Kosslyn, S. M. (2007). Neuroimaging evidence for object model verification theory: Role of prefrontal control in visual object categorization. NeuroImage,34(1), 384-398. doi:10.1016/j.neuroimage.2006.09.008

Guclu, U., & Gerven, M. A. (2015). Deep Neural Networks Reveal a Gradient in the Complexity of Neural Representations across the Ventral Stream. Journal of Neuroscience,35(27), 10005-10014. doi:10.1523/jneurosci.5023-14.2015

Khaligh-Razavi, S., & Kriegeskorte, N. (2014). Deep Supervised, but Not Unsupervised, Models May Explain IT Cortical Representation. PLoS Computational Biology,10(11). doi:10.1371/journal.pcbi.1003915

Kobatake, E., & Tanaka, K. (1994). Neuronal selectivities to complex object features in the ventral visual pathway of the macaque cerebral cortex. Journal of Neurophysiology,71(3), 856-867. doi:10.1152/jn.1994.71.3.856

Kriegeskorte, N., Mur, M., Ruff, D. A., Kiani, R., Bodurka, J., Esteky, H., . . . Bandettini, P. A. (2008). Matching Categorical Object Representations in Inferior Temporal Cortex of Man and Monkey. Neuron,60(6), 1126-1141. doi:10.1016/j.neuron.2008.10.043

Levy, B. J., & Wagner, A. D. (2011). Cognitive control and right ventrolateral prefrontal cortex: Reflexive reorienting, motor inhibition, and action updating. Annals of the New York Academy of Sciences,1224(1), 40-62. doi:10.1111/j.1749-6632.2011.05958.x

Lowe, D. G. (2004). Distinctive Image Features from Scale-Invariant Keypoints. International Journal of Computer Vision,60(2), 91-110. doi:10.1023/b:visi.0000029664.99615.94

Majaj, N., Hong, H., Solomon, E., & DiCarlo, J. (2012). A unified neuronal population code fully explains human object recognition.

Malach, R., Levy, I., & Hasson, U. (2002). The topography of high-order human object areas. Trends in Cognitive Sciences,6(4), 176-184. doi:10.1016/s1364-6613(02)01870-3

Mishkin, M., Ungerleider, L. G., & Macko, K. A. (1983). Object vision and spatial vision: Two cortical pathways. Trends in Neurosciences,6, 414-417. doi:10.1016/0166-2236(83)90190-x

Mutch, J., & Lowe, D. G. (2008). Object Class Recognition and Localization Using Sparse Features with Limited Receptive Fields. International Journal of Computer Vision,80(1), 45-57. doi

Paninski, L., Pillow, J., & Lewi, J. (2007). Statistical models for neural encoding, decoding, and optimal stimulus design. Progress in Brain Research Computational Neuroscience: Theoretical Insights into Brain Function,493-507. doi:10.1016/s0079-6123(06)65031-0

Pinto, N., Cox, D. D., & Dicarlo, J. J. (2008). Why is Real-World Visual Object Recognition Hard? PLoS Computational Biology,4(1). doi:10.1371/journal.pcbi.0040027

Riesenhuber, M., & Poggio, T. (1999). Hierarchical models of object recognition in cortex. Nature Neuroscience,2(11), 1019-1025. doi:10.1038/14819

Safavi, S., Dwarakanath, A., Kapoor, V., Werner, J., Hatsopoulos, N. G., Logothetis, N. K., & Panagiotaropoulos, T. I. (2018). Nonmonotonic spatial structure of interneuronal correlations in prefrontal microcircuits. Proceedings of the National Academy of Sciences,115(15). doi:10.1073/pnas.1802356115

Scalaidhe, S. P. (1999). Face-selective Neurons During Passive Viewing and Working Memory Performance of Rhesus Monkeys: Evidence for Intrinsic Specialization of Neuronal Coding. Cerebral Cortex,9(5), 459-475. doi:10.1093/cercor/9.5.459

Serre, T., Oliva, A., & Poggio, T. (2007). A feedforward architecture accounts for rapid categorization. Proceedings of the National Academy of Sciences,104(15), 6424-6429. doi:10.1073/pnas.0700622104

Sharpee, T. O., Kouh, M., & Reynolds, J. H. (2013, July 09). Trade-off between curvature tuning and position invariance in visual area V4. Retrieved from http://www.pnas.org/content/110/28/11618.short

Tiny ImageNet. (2017). Retrieved from https://www.kaggle.com/c/tiny-imagenet
Ullman, S. (1997). High-level vision: Object recognition and visual cognition. Cambridge, MA: The MIT Press.

Wistuba, M., Schilling, N., & Schmidt-Thieme, L. (2015). Hyperparameter Search Space Pruning – A New Component for Sequential Model-Based Hyperparameter Optimization. Machine Learning and Knowledge Discovery in Databases Lecture Notes in Computer Science,104-119. doi:10.1007/978-3-319-23525-7_7

Yamins, D. L., Cadieu, C. F., Solomon, E. A., Seibert, D., & DiCarlo, J. J. (2014, June 10). Performance-optimized hierarchical models predict neural responses in higher visual cortex. Retrieved from https://doi.org/10.1073/pnas.1403112111

Yamins, D. L., & Dicarlo, J. J. (2016). Using goal-driven deep learning models to understand sensory cortex. Nature Neuroscience,19(3), 356-365. doi:10.1038/nn.4244

Yau, J. M., Pasupathy, A., Brincat, S. L., & Connor, C. E. (2012). Curvature Processing Dynamics in Macaque Area V4. Cerebral Cortex,23(1), 198-209. doi:10.1093/cercor/bhs004