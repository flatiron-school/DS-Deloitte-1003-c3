# Aggregating and Combining DataFrames

The main goals here are to introduce more pandas tools like `.groupby()`, `.pivot()` and `.merge()`, and then give students some practice in analysis with the Animal Center dataset.

Materials:
- [Jupyter Notebook: Aggregating and Combining DataFrames](aggregating_combining_dataframes.ipynb)

## Learning Goals

- Use GroupBy objects to organize and aggregate data
- Create pivot tables from DataFrames
- Combine DataFrames by merging, joining, and concatinating

## Lesson Plan - 1 hour 15 minutes

### Introduction (5 minutes)

More advanced tools in pandas! Also - while we'll start out with just the Outcomes data, we'll be using both the Intakes and Outcomes data from the same Austin Animal Center (used in both Pandas DataFrames and Pandas Data Cleaning)

### .groupby() (20 minutes)

Used for aggregating. This will be paralleled precisely by SQL syntax.

There's an activity here that should take students about 5 minutes, then come back together - students might not realize that, since we read in the Date of Birth column as a datetime object, `.max()` will give the most recent date - discuss.

### .pivot_table() (10 minutes)

Like Excel. But not all students will be familiar with this.

### Discussion of .merge(), .join(), .concat() (20 minutes)

Pull up the docstrings for these to make sure students understand the proper syntax here. Note that `df1.merge(df2)` and `pd.merge(df1, df2)` are both possible.

Note that `.merge()` and `.join()` are quite similar, but the latter joins on indices.

### Back to the Animal Center (15 minutes)

Key here is that, since we've done nothing with duplicate animal IDs, the join (even inner joins) are blowing up the table. Good time to showcase dealing with duplicates, since that's not a part of the data cleaning notebook at the moment. 

### Conclusion (5 minutes)

Can discuss the level up pieces, which are just some ways to make accessing pandas data a little easier.