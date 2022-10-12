# Data Visualization with Matplotlib

Overall goal is to learn what makes an effective visualization using different `matplotlib`-based syntaxes.

Materials:
- [Jupyter Notebook: Plotting with Pandas](plotting-with-pandas.ipynb)

## Learning Goals

- Understand why we visualize data
- Recognize the different matplotlib-based ways we can create visualizations
- Choose appropriate graph types based on data and problem context for explanatory visualizations
- Use matplotlib syntax to adjust elements and improve visualizations
- Create figures containing multiple graphs

## Lesson Plan - 1 hour 30 minutes

(Removing the exercises, including around exploring plot types and improving visualizations, should keep this lecture to about an hour)

### Visaulization Motivation (15 minutes)

In the first section, allow students to think critically about what information "jumps out" to them (DataFrame or descriptive statistics or visualization)

Then, discuss the different reasons they might make visualizations - specifically, exploratory vs explanatory plots. 

### Plotting Syntax (15 minutes)

Showcase 4 different approaches - using a simple pandas `.plot` method, then using Seaborn, then using matplotlib's `.plt`, then using matplotlib's `subplots`. Point out that these increase in complexity - the last one gives students the most control over elements in their plot, and thus is ideal for crafting explanatory visualizations they would show to others. Also bring up that these things can often stack and interact, because all of these are built on top of matplotlib.

### Exploring Plot Types (15 minutes)

Discuss - and definitely click into the Python Graph Gallery and showcase how to use that as a resource.

For specifics about which plot to use when, or how to build specific plots, there are lots of resources both in the curriculum and online (and, for the latter, in the documentation).

There's an exercise, with three scenarios to discuss which visual might work best - shouldn't take more than 10 minutes.

### Improving Visualizations (35 minutes)

This should take the most amount of time and focus in this lecture.

Spend a few minutes discussing accessible visualizations - the article linked is a great resource for thinking about accessibility in this context.

Then, spend about 10 minutes stepping through and changing really specific pieces, like colors and sizes. Showcase how easy it can be to add callout text/annotations to a plot for emphasis. Be sure to showcase how to properly save a visual from a notebook as an image file.

Let students take about 5 minutes to improve the visualization of the most common animal types (from the Austin Animal Center data, same as the Pandas DataFrame lecture), then discuss a few improvements.

### Multiple Plots (5 minutes)

Show how multiple plots can be made on one figure emphasizing the different Axis objects returned by `subplots()`. Typically you'll walk through this quickly, just running the code and discussing when this might be useful.

### Conclusion (5 minutes)

There's a lot of written Level Up content at the end - encourage students to read through on their own. Then there's stuff about changing general plot styles which can be a nice touch.