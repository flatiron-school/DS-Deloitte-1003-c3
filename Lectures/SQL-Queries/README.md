# SQL Queries

Expose students to SQLite and SQL SELECT syntax, as well as to the structure of relational databases.

Materials:
- [Jupyter Notebook: SQL Queries](sql_queries.ipynb)

## Before Lecture Prep

Slack the following message to students a few hours before lecture:

> Looking forward to diving into SQL with you all soon! To prepare for our first SQL lecture, please read Peter Bell's great piece [on relational databases](https://flatironschool.com/blog/an-introduction-to-the-relational-database).

## Learning Goals

- Describe relational databases
- Connect to a SQLite database and get schema information
- Use SQL SELECT and `pd.read_sql()` to query databases 
- Use WHERE, ORDER BY and LIMIT to modify queries

## Lesson Plan

### Intro (5 Mins)

Reinforce how much grads have said this is important, that this will almost certainly be used in their jobs and tested in interviews. Can discuss of the role SQL plays in the DS ecosystem. Association of SQL with database design / maintenance, ETL, data engineering. Refer to Peter Bell's piece to motivate the value of relational data tables in terms of space, cost, redundancy.
 
### Relational Databases (5 Mins)

Give students a quick visual preview of what they will be working with. 

### SQLite (10 Mins)

Get students to the point that they can get and interpret a schema. Use `pd.read_sql()` for easy reading of results. When describing SQLite, contrast it with more traditional models that have a server-client structure.

### SQL SELECT (20 Mins)

Demoing queries with flights.db, adding new clauses as you go. Reinforce the structure of the SQL query, that you are showing them the main pieces they'll need.

### Exercises + Wrap Up (20 Mins)

Have students work on these individually or in groups, then discuss answers as a whole class. 

## Lecture Supplements

[Netflix has a great article](https://medium.com/netflix-techblog/notebook-innovation-591ee3221233) describing three different data roles at their company, their different needs, and their toolsets.

[AI Hierarchy of Needs](https://hackernoon.com/the-ai-hierarchy-of-needs-18f111fcc007)
