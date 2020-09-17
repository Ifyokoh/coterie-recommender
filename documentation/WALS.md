# Weighted Alternating Least Squares (WALS) Engine in Recommendation System

## Table of Contents
- [Introduction](#intro)
- [Weighted alternating least squares(WALS)](#wals)
- [Why is WALS used in recommendation system](#why)
- [How does WALS  work](#how)

## Introduction<a name="intro"></a>
Recommendation systems are composed of algorithms that aim to predict the preference a user would assign to a given item. Collaborative filtering algorithm has an advantage among other recommendation algorithms because it is capable of recommending complex items without a prior knowledge about the item. It has two primary approaches: memory-based and model based approaches.
- **Memory Based:** This approach takes user rating data to calculate similarities between users and items and then produce a prediction for the user by taking the weighted average of all ratings. A common algorithm here is K-nearest neighbor (KNN), while KNN is simple to compose, the performance degrades as the data becomes sparse because in a real world setting, the vast majority of products receive very few or even no ratings at all by users and this causes an extremely sparse matrix with more than 99% of entries missing.
- **Model Based:** This approach handles the sparsity in the data by building models to discover patterns in the training data and then use the model to make predictions on the real data. Model based approach uncovers latent factors which can be used to construct the training data ratings.

## Weighted alternating least squares(WALS)<a name="wals"></a>
Alternating Least Squares(ALS) is one of the model based methods used to tackle collaborative filtering with implicit feedback, it that rotates between fixing one of the unknowns i.e. fixing the latent factor of the item and then in the next iteration fixing the latent factor of the user. When one is fixed the other can be computed by solving the least-squares problem. This approach is useful because when either the user-factors or the item factors are fixed, the cost function becomes quadratic so its global minimum can be readily computed.

[Weighted alternating least squares](https://github.com/GoogleCloudPlatform/tensorflow-recommendation-wals/tree/master/wals_ml_engine) is an algorithm very similar to ALS but different in a way that unobserved values are assigned different weights for unobserved entries and observed entries.  WALS is aimed at trying to optimize collaborative filtering algorithms for datasets that are derived off of implicit ratings

## Why is WALS used in recommendation system<a name="why"></a>
There are two kinds of data available where alternative least square algorithm can be applied:
- Explicit feedback: A score, such as a rating or a like
- Implicit feedback: Not as obvious in terms of preference, such as a click, view, or purchase

In implicit feedback, instead of asking customers to tell you what they think about your product or how they rate them, it is much easier to observe user behavior. Also we need to consider that a user not taking any positive action on an item can stem from many other reasons beyond not liking it. For example, the user might be unaware of the existence of the item, or unable to consume it due to its price or limited availability or might not like an item but purchased it as a gift for someone else. Thus, the need to have different confidence levels among items that are indicated to be preferred by the user, a stronger indication that a user indeed likes the item can only be measured by how high the confidence grows.  

WALS is needed in this case as it optimizes datasets that are derived off of implicit ratings and also performs faster in large datasets due to the fact that it is easily parallelized. WALS is implemented in Apache Spark ML and built for large-scale collaborative filtering problems. ALS is doing a pretty good job at solving scalability and sparseness of the Ratings data, and it’s simple and scales well to very large datasets.

## How does WALS  work<a name="how"></a>
The WALS matrix factorization algorithms work by decomposing the user-item interaction matrix into the product of two lower dimensionality rectangular matrices. One matrix can be seen as the user matrix where rows represent users and columns are latent factors. The other matrix is the item matrix where rows are latent factors and columns represent items.

 ![Image](https://i.imgur.com/85WgSJg.png)

Assume we have a matrix R of size MxN, where M is the number of users and N is the number of items. This matrix is quite sparse, since most users only interact with a few items each. We can factorize this matrix into two separate smaller matrices: one with dimensions MxK which will be our latent user feature vectors for each user (U) and a second with dimensions KxN, which will have our latent item feature vectors for each item (I). Multiplying these two feature matrices together approximates the original matrix R, but now we have two matrices that are dense including a number of latent features K for each of our items and users. In order to solve for U and I, we apply ALS to approximate it. We only need to solve one feature vector at a time, which means it can be run in parallel. To do this, randomly initialize U and solve for I. Then go back and solve for U using the solution for I. Keep iterating back and forth like this until the best convergence that approximates R is gotten.

## References
https://towardsdatascience.com/prototyping-a-recommender-system-step-by-step-part-2-alternating-least-square-als-matrix-4a76c58714a1

http://yifanhu.net/PUB/cf.pdf

https://jessesw.com/Rec-System/  

