# Pandas Data Cleaning

The goal here is to cover more sophisticated bits of Python syntax focused around data cleaning.

Materials: 
- [Jupyter Notebook: Pandas Data Cleaning](pandas_data_cleaning.ipynb)

## Learning Goals

- Handle missing data, and recognize when different strategies for handling missing data would be appropriate
- Use DataFrame methods (and sometimes lambda functions) to transform data
- Use string methods to transform object-type columns

# Lesson Plan - 1 hour 15 minutes

### Introduction (5 minutes))

Introduce the lecture goals and the dataset - same Austin Animal Center but a different dataset, as this one is Outcomes data (not Intakes, which is used in the first Pandas lecture)

### Question 1: How old are the animals in our dataset (in a consistent unit, like days)? (30 minutes)

Note that there's a DateTime column (presumably, the DateTime of the logged outcome) and a Date of Birth column - yes, you can calculate the age of the animals that way, and we will by the end of this section! But we're using the Age upon Outcome column to teach string methods, showcasing the different ways we can access and transform data. 

This approach facilitates our conversation of null values - here, the focus is on thinking both WHY the data might be missing, as well as the best way to fill the data for the specific purpose we have in mind.

This approach also facilitates the discussion around `map`, `apply`, and `applymap` - here, we'll use a `map` function to transform the age values using the provided dictionary, and later an `apply` function will be used.

### Question 2: Are most of the animals already fixed? (20 minutes)

Here, we can use either `apply` or `map` to make our transformation. 

### Lambda Functions (10 minutes)

Showcase filling nulls with both a lambda and `.fillna`

### Building Indicator Columns (5 minutes)

Showcase how it's easy to build an indicator column for different traits, which can be used to find patterns.

### Conclusion (5 minutes)

Might briefly go over the level up portions about `applymap` and faster numpy methods.
