# SQL Joins and Aggregation

Materials:
- [Jupyter Notebook: SQL](sql.ipynb)

## Learning Goals

- Use SQL aggregation functions with GROUP BY
- Use HAVING for group filtering
- Use SQL JOIN to combine tables using keys

## Lesson Plan 

### Introduction (5 Mins)

We need new syntax to exploit the relational structure of our database. Hence JOINs. The GROUP BY syntax will work much like the `pandas` analogue.

### GROUP BY statements (15 Mins)

Motivating the GROUP BY construction by doing the same work with a different set of queries (that are multiple, longer, etc.).

The answer to the exercise includes an ORDER BY ... DESC clause to show the top counts after the aggregation.

### Filtering Groups with HAVING (15 Mins)

Like WHERE but for groups!

### JOINs (20 Mins)

Note that SQLite supports only LEFT JOINs, INNER JOINs, and CROSS JOINs (not RIGHT or FULL OUTER JOINs).

### Conclusion (5 Mins)

One more syntactic structure to look at––subqueries. We also want to step back and talk about the many forms/platforms/dialects in which one can encounter SQL.

## Tips

With both aggregations and JOINs aliases become more important. Make sure to cover how those are working in the various examples and exercises.