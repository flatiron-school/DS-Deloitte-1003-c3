# Big Data Introduction

## Learning Goals

- Understand what is "big data"
- Know why big data is different and how it can be processed
- Understand how data can be handled in distributed and parallel systems
- Understand how MapReduce is run

## Lesson Materials

- [Jupyter Notebook: Big Data Intro](big_data_intro.ipynb)

## Lesson Plan

### Introduction (10 Mins)

- Discuss what makes "big data" different and hence the need for big data tools
- "Numbers everyone should know"
- Three 'V's of big data

### Different Tools for Big Data (15 Mins) 

- Hadoop and basics of what it does ("old school" but still used)
- Spark replacing Hadoop and deeper discussion. Aside sections for more information as the lecturer decides
- Discuss that it's not just one tool; different tools for different tasks

### Parallel and Distributed Systems (10 Mins)

- Why and how we use parallel and distributed systems
- Can use different analogies to explain how the shared resources help speed up certain tasks/jobs

### MapReduce Explanation (10 Mins)

- Explain the steps of MapReduce and how it relates to distributed & parallel systems.

## MapReduce Implementation (15 Mins)

- Implement MapReduce example with [`mrjob`](https://mrjob.readthedocs.io/en/latest/index.html)
- Focus on the concepts of MapReduce over the exact syntax
- Note: This can be turned into an exercise by introducing a new example and/or taking the second example ("Word Count")

