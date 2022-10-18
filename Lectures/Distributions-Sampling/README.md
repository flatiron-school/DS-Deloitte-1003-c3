# Distributions and Sampling

This lecture introduces PMFs, PDFs, CDFs, with a short activity at the end

## Learning Goals

* Explain how sampling is related to its related population
* Describe the difference between discrete and continuous random variables
* Describe the difference between PMFs, PDFs, and CDFs

## Lecture Materials

[Jupyter Notebook: Distributions and Sampling](Distributions-Sampling.ipynb)

## Lesson Plan - 1 hour 30 minutes

### Sampling (15 minutes)

Discuss how we sample from a population to motivate why it's useful to know how to do probability distributions.

Introduce the Seattle Employees dataset, and use that to demonstrate how adjusting sample size leads to a more or less representative sample.

### Probability Distributions & Probability Distribution Functions (15 minutes)

Discuss how probability distributions are foundational to all statistics. Discuss how distributions describe the underlying patterns - what we see in datasets is then representative of the underlying distribution. Discuss the difference between continuous and discrete distributions, and why it matters.

High level discussion of probability distribution functions - how they describe and relate different aspects of distributions, and allow us to answer questions.

### Distributions with Scipy's Stats Module (25 minutes)

Walk through different kinds of distributions (bernoulli, uniform, then normal) to showcase how to use the `stats module` in python. Goal is for students to recognize which arguments each of the distributions need and to start building familiarity with the module and its documentation, so they can continue using it in the future.

### Examples in Python (15 minutes)

As a big group, showcase a few examples of how you can answer questions using the `stats` module. The visualizations should help solidify how exactly the probability distribution functions allow us to answer those questions.

### Exercises (15 minutes)

If you have time, give students 5-10 minutes to answer the order totals exercise on their own or in breakout rooms. Then come together and discuss.

### Conclusion (5 minutes)

Close out by discussing how we describe distributions. Note that there are extensive level up sections that students can explore on their own.