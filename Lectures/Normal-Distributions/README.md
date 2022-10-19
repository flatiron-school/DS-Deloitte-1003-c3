# Normal Distributions

## Learning Goals

- Describe the normal distribution's Empirical Rule
- Calculate z-scores from a normal distribution through standardization
- Define the Central Limit Theorem 

## Lecture Materials

[Jupyter Notebook: Normal Distributions](NormalDistributions.ipynb)

## Lesson Plan - 1 hour

### Intro - Motivating the Normal Distribution (5 minutes)

Why the normal distribution is so important and relevant.

### Standard Normal Distribution (5 minutes)

Coming from the normal distribution, this section connects the previous section with the next one on z-scores. May want to foreshadow sklearn's `StandardScaler` here, and that using standardized data can allow for comparisons among data that are on different scales.

### Z-Score (10 minutes)

Discuss the formula for a z-score - how all you're doing by calculating the z-score for a specific point is changing the point from the units of the original distribution to the units of standard deviations. Might highlight that doing this for all points would standardize them, just like above, but z-scores are calculated for individual points and typically thought of in those more specific terms.

There's a site linked here that lets you build and interact with distributions which can be fun if you have time.

### Empirical Rule (5 minutes)

Introduce this as something people often have memorized, as a nice property of normal distributions.

### Exercises (15 minutes)

Recommend splitting students into breakout rooms and giving them about ten minutes to use the normal distribution details provided and answer these 4 questions (4 including the bonus). Then come back together and discuss answers and showcase different ways to arrive at these solutions.

### Z-Scores Using Pandas (10 minutes)

Goes back to the Seattle wage data introduced in the Distributions Intro lecture and showcases how to calculate a z-score for one specific row. Then it standardizes the whole column and visualizes how the distribution does (and doesn't) look different after standardization. 

**Note:** be sure to toggle the `sharex` value, running it as both `True` and `False`, to showcase how the underlying distribution doesn't change but the scale does!

### Central Limit Theorem (10 minutes)

The above distribution isn't really normal, so it's a natural way to connect to the Central Limit Theorem. This introduction does not need to be thorough, but students should grasp that taking repeated samples from a non-normal distribution allows us to draw inferences about the underlying population parameters. Do not need to spend a lot of time on this, but have some visuals to explain and explore.

### Conclusion (5 minutes)

There are several level up sections about other distributions, and students might be interested in explorign the distributions type exercise in the level up section. There's also a preview of sklearn's `StandardScaler`.