# Structured Query Language SQL:
Talks with the databases like:
* SQL Server
* Oracle
* MySQL, etc.

Tools to Write SQL
* SQL Server Management Studio
* SQL Workbench
* SQL developer
* TOAD, etc.

A table in SQL is a type of entity (i.e. Dogs), and each row in that table as a specific *instance* of that type (i.e. A pug, a beagle, a different colored pug, etc.). This means that the columns would then represent the common properties shared by all instances of that entity (i.e. Color of fur, length of tail, etc.)

## `SELECT` queries:
`SELECT` is used to retrieve data from SQL database, and it is also colloquially called as *queries*. Queries contains mainly -> what data we are looking for, where to find it in the database, and optionally transform it before returning.
```sql
/* Select query for a specific multiple columns */
SELECT column, another_col
FROM myTable;

/* to select all the columns from a table */
SELECT *
FROM myTable;
```

## Queries with Constraints `WHERE` clause:
The `WHERE` clause is applied to each row of data by checking specific column values to determine whether it should be included in the results or not.
```sql
/* Select query with constraints */
SELECT col1, col2
FROM myTable
WHERE condition
    AND/OR another_condition
    AND/OR ....;
    
/* example */
SELECT *
FROM myTable
WHERE col2='value';
```
Some more complex queries can be made by combining several of ANDs and ORs as shown below:

<p align="center">
    <img src="./images/img1.png" alt="sql joins types" width="700" />
</p>


When doing `WHERE` clauses with the columns containing text data, we have several more sql commands to do selection of the values in the column that works with the strings. See below:

<p align="center">
    <img src="./images/img2.png" alt="sql joins types" width="700" />
</p>


## Filtering and Sorting Query Results:
`DISTINCT` keyword will discard rows that have a duplicate column value.
```sql
/* removes every rows for col1, col2 has same values, need to discard duplicates based on
specific columns then we need to use GROUP BY clause */
SELECT DISTINCT col1, col2
FROM myTable
WHERE condition(s);
```

### Ordering results with `ORDER BY`:
```sql
SELECT col1, col2
FROM myTable
WHERE condition(s)
ORDER BY col1 ASC/DESC;
```
When an `ORDER BY` clause is specified, each row is sorted alpha-numerically based on the specified column's value.

### Limiting results to a subset using `LIMIT` and `OFFSET`:
The `LIMIT` wll reduce the number of rows to return, and the optional `OFFSET` will specify where to begin counting the number of rows from.
```sql
SELECT col1, col2
FROM myTable
WHERE condition(s)
ORDER BY col1 ASC/DESC
LIMIT num_limit OFFSET num_offset;
```

## Multi-table queries with `JOINS`:
Of course in the real world, we don't just have the single table, but data closely related are spread across several tables, and we should be able to access from multiple tables.

Tables that share information about a single entity need to have a *primary key* that identifies that entity *uniquely* across the database.

### Let's see **INNER JOIN** type of join:
```sql
/* select query with INNER JOIN on multiple tables */
SELECT col1, another_table_col2, ...
FROM myTable
INNER JOIN anotherTable
    ON myTable.id = anotherTable.id
WHERE condition(s)
ORDER BY col, ... ASC/DESC
LIMIT num_limit OFFSET num_offset;
```

### OUTER JOINS:
In the inner joins, resulting table only contains data that belongs in both of the tables. But, most of the time, data won't be symmetric, in that case, in order to not lose data from any of the joined tables, we need to use either `LEFT JOIN`, `RIGHT JOIN`, or `FULL JOIN`.
```sql
/* Select query with LEFT/RIGHT/FULL JOINs on multiple tables */
SELECT col1, another_table_col, ...
FROM myTable
INNER/LEFT/RIGHT/FULL JOIN another_table
    ON myTable.id = another_table.matching_id
WHERE condition(s)
ORDER BY col, ... ASC/DESC
LIMIT num_limit OFFSET num_offset;
```

See image below for more on joins:

<p align="center">
    <img src="./images/sql_joins.jpeg" alt="sql joins types" width="700" />
</p>











## Resources:

* [SQLBolt](https://sqlbolt.com/)