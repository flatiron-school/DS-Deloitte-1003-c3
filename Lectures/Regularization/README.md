# Regularization

## Learning Goals

- explain the notion of "validation data";
- explain and use the algorithm of cross-validation;
- explain the concept of regularization;
- use Lasso and Ridge regularization in model design.

## Lecture Materials

[Jupyter Notebook: Regularization Lecture](notebooks/Regularization_Lecture.ipynb)

## Lesson Plan

Provide a breakdown of the lecture activities using the structure below. 

### Introduction (5 Mins)

Regularization is all about combatting overfitting, so we start by reviewing bias vs. variance.

### Review of Validation and Cross-Validation (15 Mins)

These concepts are introduced at the end of Phase 2, but there's often not much time for them then.

### Regulatization Through Modification of Cost Function (15 Mins)

Regularization in a broad sense is any strategy to reduce overfitting, but the new idea here is to introduce terms into the loss  function that are proportional to the coefficients. We could minimize the absolute values of the betas (Lasso) or their squares (Ridge). The Elastic Net is a hybrid with a percentage of each.

### Ridge/Lasso in Python (10 Mins)

The example uses third-degree polynomials to produce an overfit model, then corrects it with Ridge.

### Choosing Regularization Strength (10 Mins)

The Greek letter lambda is generally used for regularization strength, but `sklearn` uses alpha. This is a tunable hyperparameter. The lecture walks through using cross-validation to find an optimal value.

### Conclusion (5 Mins)

Cross-validation raises the issue of data leakage: info from dev or test creeping into our training data.

## Tips

This runs a bit on the long side. But the cross-val material should be review, and so I (Greg) try to go quickly through that.

## Lecture Source Materials

* [_Python DataScience Handbook_ Chapter 5 on Linear Regression (Regularization)](https://jakevdp.github.io/PythonDataScienceHandbook/05.06-linear-regression.html#Regularization)
