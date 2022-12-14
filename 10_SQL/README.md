# Structured Query Language SQL:
It is also pronounced as "SEQUEL". It talks with the relational databases like:
* Microsoft SQL Server
* PostgreSQL
* Oracle
* SQLite
* MySQL, etc.

It can read data from the RDBMS (relation database) or write data into database. SQL can also create,delete modify database. In RDMS system, different tables can talk to each other relating them together using the common column values present in two table, for example.

Some of the tools to Write and practice SQL:
* SQL Server Management Studio
* SQL Workbench
* SQL developer
* TOAD, etc.
* Online tool to practice: [phpMyAdmin](https://www.phpmyadmin.net/)

> :bulb: **Caution**: Every database system has their own way to writing many of the syntax/queries that we see below. Only few syntax are standard, in that they are written same way in all platforms. So, you are always advised to look into the official documentation for each of the platforms (i.e. postgres, mysql, sqlite, microsoft sql, etc.) to know the specific syntax. Having said that, however, all the platforms do support the following functionality, and it's just syntax varies. Once you know certain functionality exist, it is however, very easy to look for in internet.

SQL commands, overall can be categorized into **5 types**. See below for the image:

<p align="center">
    <img src="./images/sql_commands.png" alt="sql joins types" width="700" />
</p>

**Constraints on data types**: When putting the data into table, we can constraints the data types and the values they can store in the column of a table. By doing this, our table will contain much cleaner data in them. Let's look at some of the contraints that we can have for a column of a table in RDMS.

<p align="center">
    <img src="./images/sql_constraints.png" alt="sql joins types" width="700" />
</p>

A table in SQL is a type of entity (i.e. Dogs), and each row in that table as a specific *instance* of that type (i.e. A pug, a beagle, a different colored pug, etc.). This means that the columns would then represent the common properties shared by all instances of that entity (i.e. Color of fur, length of tail, etc.)

## quick mysql server status commands
```bash
sudo service mysql status   # to check the status
sudo service mysql stop     # to stop mysql service
sudo service mysql start    # to start server
sudo service mysql restart  # to re-start server
```

## Create Database:
```sql
CREATE DATABASE name;    /* this command creates database */
USE name;                /* USE activates the database */
```

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

## Queries with Constraints `WHERE` clause, `LIKE`, and `IN`:
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

/* using IN (same as above) */
SELECT *
FROM myTable
WHERE col2 IN ('value');

SELECT *
FROM myTable
WHERE col2 NOT IN ('value'); /* everything other that one value from col2 */

/* real examples */
SELECT * FROM Country;
SELECT Name, Continent, Population FROM Country 
  WHERE Population < 100000 ORDER BY Population DESC;
  
SELECT Name, Continent, Population FROM Country 
  WHERE Population < 100000 OR Population IS NULL ORDER BY Population DESC;
  
SELECT Name, Continent, Population FROM Country 
  WHERE Population < 100000 AND Continent = 'Oceania' ORDER BY Population DESC;
  
/* % wild card anything before and after is okay */
/* _a% first letter can be anything, second a and anything after */
SELECT Name, Continent, Population FROM Country 
  WHERE Name LIKE '%island%' ORDER BY Name;
  
/* same like .isin() in python */
SELECT Name, Continent, Population FROM Country 
 WHERE Continent IN ( 'Europe', 'Asia' ) ORDER BY Name;
```
Some more complex queries can be made by combining several of ANDs and ORs as shown below:

<p align="center">
    <img src="./images/img1.png" alt="sql joins types" width="700" />
</p>


When doing `WHERE` clauses with the columns containing text data, we have several more sql commands to do selection of the values in the column that works with the strings. See below:

<p align="center">
    <img src="./images/img2.png" alt="sql joins types" width="700" />
</p>


## Conditional expressions when querying:
Sometimes we want to query if a col value is one thing or another, and do something based on conditions. Just like if else statements in python.
```sql
CREATE TABLE booltest (a INTEGER, b INTEGER);
INSERT INTO booltest VALUES (1, 0);
SELECT * FROM booltest;

/* testing if a ,b is either True of False and asking to print corresponding string */
SELECT
    CASE WHEN a THEN 'true' ELSE 'false' END as boolA,
    CASE WHEN b THEN 'true' ELSE 'false' END as boolB
    FROM booltest
;

/* testing if a, b is specific values or not */
SELECT
  CASE a WHEN 1 THEN 'true' ELSE 'false' END AS boolA,
  CASE b WHEN 1 THEN 'true' ELSE 'false' END AS boolB 
  FROM booltest
;

/* case is like if else statements, see example */
SELECT staff_id, salary,
CASE WHEN salary >= 10000 THEN 'High Salary'
    WHEN salary BETWEEN 5000 AND 9999 THEN 'Average Salary'
    WHEN salary < 5000 THEN 'Too low Salary'
END AS RANGE
FROM staff_salary
ORDER BY 2 DESC;
```

## Subqueries in `WHERE` Clause:
Subqueries are put in the `WHERE` clause, and they are a nested `SELECT` statements and are quite powerful tool to know.
```sql
/* IDs and names of students who have applied to major in CS at some college */
select sID, sName
from Student
where sID in (select sID from Apply where major='CS')

/* Students who have applied to major in CS but have not applied to major in EE */
select sID, sName
from Student
where sID in (select sID from Apply where major='CS')
    and sID not in (select sID from Apply where major='EE')
```

## Subqueries in `FROM` Clause:
```sql
select *
from (select sID, sName, GPA, GPA*(sizeHS/1000.0) as scaledGPA 
      from student) G
where abs(G.scaledGPA - GPA) > 1.0;
```

## Subqueries in `SELECT` Clause:
```sql
/* Colleges paired with the highest GPA of their applicants */
/* the result of subqueries in the select should only return one value at a time  */
select cName, state,
(select distinct GPA
from College, Apply, Student
where College.cName = Apply.cName
    and Apply.sID = Student.sID
    and GPA >= all
            (select GPA from Student, Apply
            where Student.sID = Apply.sID
                and Apply.cName = College.cName)) as GPA
from college;
```


## Filtering and Sorting Query Results:
`DISTINCT` keyword will discard rows that have a duplicate column value.
```sql
/* removes every rows for col1, col2 has same values, need to discard duplicates based on
specific columns then we need to use GROUP BY clause */
SELECT DISTINCT col1, col2
FROM myTable
WHERE condition(s);

/* examples */
SELECT Continent FROM Country;
SELECT DISTINCT Continent FROM Country; /* finds all distinct values in Continent col */

SELECT DISTINCT a,b FROM test; /* shows distinct combinations of a & b col values */
```

### Ordering results with `ORDER BY`:
```sql
SELECT col1, col2
FROM myTable
WHERE condition(s)
ORDER BY col1 ASC/DESC;

/* examples */
SELECT Name FROM Country;
SELECT Name FROM Country ORDER BY Name;
SELECT Name FROM Country ORDER BY Name DESC;
SELECT Name FROM Country ORDER BY Name ASC;
SELECT Name, Continent FROM Country ORDER BY Continent, Name; /* first order by Continent, then Name */
/* here, we order DESC by Continent, then Region, Name ASC (by default) */
SELECT Name, Continent, Region FROM Country ORDER BY Continent DESC, Region, Name;
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

## Counting rows:
Instead of listing the output/result of the query as tables, which is what we mostly want to do anyway, we can also `COUNT` the rows/instances of our queries:
```sql
SELECT COUNT(*) FROM myTable;       /* counts total rows in myTable */
SELECT COUNT(*) FROM Country WHERE Population > 100000000 AND Continent = 'Europe'; /* count with conditions */
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

/* examples */
/* l,r are an alias we can define later in second line */
SELECT l.description AS left, r.description AS right /* since both col same name, so we give alias */
  FROM left AS l
  JOIN right AS r ON l.id = r.id;
  
SELECT s.id AS sale, i.name, s.price 
  FROM sale AS s
  JOIN item AS i ON s.item_id = i.id
  ;

SELECT s.id AS sale, s.date, i.name, i.description, s.price 
  FROM sale AS s
  JOIN item AS i ON s.item_id = i.id
  ;
  
/* sometimes, we need to query data from more than 2 tables */
/* let's say we have 3 tables, Junction Table */
SELECT * FROM customer;
SELECT * FROM item;
SELECT * FROM sale;

SELECT c.name AS Cust, c.zip, i.name AS Item, i.description, s.quantity AS Quan, s.price AS Price
  FROM sale AS s
  JOIN item AS i ON s.item_id = i.id
  JOIN customer AS c ON s.customer_id = c.id
  ORDER BY Cust, Item
;

-- a customer without sales
INSERT INTO customer ( name ) VALUES ( 'Jane Smith' );
SELECT * FROM customer;

-- left joins
SELECT c.name AS Cust, c.zip, i.name AS Item, i.description, s.quantity AS Quan, s.price AS Price
  FROM customer AS c
  LEFT JOIN sale AS s ON s.customer_id = c.id
  LEFT JOIN item AS i ON s.item_id = i.id
  ORDER BY Cust, Item
;
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

Supported Types of Joins in **MySQL**
* `INNER JOIN`: returns records that have matching values in both tables
* `LEFT JOIN`: returns all records from the left table, and the matched records from the right table
* `RIGHT JOIN`: returns all records from the right table, and the matched records from the left table
* `CROSS JOIN`: this is what an OUTER JOIN above looks like, complete records from both tables

See image below for more on joins:

<p align="center">
    <img src="./images/sql_joins.jpeg" alt="sql joins types" width="700" />
</p>

## Dealing with `NULL`:
`NULL` values arises in several cases, when we try to outer join two asymmetric tables, for example. We can test whether a column contains any `NULL` values or not using `WHERE` clause together with `IS NULL` or `IS NOT NULL` constraints.
```sql
/* Select query with constraints on NULL values */
SELECT col1, col2, ...
FROM myTable
WHERE col1 IS/IS NOT NULL
AND/OR another_condition
AND/OR ...;
```

## Queries with mathematical/logical expressions:
In addition to doing simple raw column queries, if possible (if values in col permits), we can also add complex mathematical and other logical expressions to make our query even more interesting.
```sql
/* see example of this physics db */
SELECT particle_speed / 2.0 AS half_particle_speed
FROM physics_table
WHERE ABS(particle_position) * 10.0 > 500.0;
```

## Queries with **aggregates**:
Aggregates functions are used to summarize the columns or group of rows of data. Its syntax looks like:
```sql
/* select query with aggregate functions over all rows */
SELECT AGG_FUNC(col_or_expr) AS agg_description, ...
FROM myTable
WHERE contraint;

/* with grouped aggregate: apply agg func to individual group */
SELECT AGG_FUNC(col_or_expr) AS agg_description, ...
FROM myTable
WHERE constraint
GROUP BY col;

/* real examples */
/* using COUNT to count all rows in Country table */
SELECT COUNT(*) FROM Country;

/* first groupby and then count */
SELECT Region, COUNT(*) AS Count
  FROM Country
  GROUP BY Region
  ORDER BY Count DESC, Region
;

SELECT a.title AS Album, COUNT(t.track_number) as Tracks
  FROM track AS t
  JOIN album AS a
    ON a.id = t.album_id
  GROUP BY a.id
  ORDER BY Tracks DESC, Album
;

SELECT COUNT(*) FROM Country;
SELECT COUNT(Population) FROM Country;
SELECT AVG(Population) FROM Country;
SELECT Region, AVG(Population) FROM Country GROUP BY Region;
SELECT Region, MIN(Population), MAX(Population) FROM Country GROUP BY Region;
SELECT Region, SUM(Population) FROM Country GROUP BY Region;

SELECT COUNT(HeadOfState) FROM Country;
SELECT HeadOfState FROM Country ORDER BY HeadOfState;
SELECT COUNT(DISTINCT HeadOfState) FROM Country; /* removing duplicates in col, then count */
```
Some of the common aggregate functions are:

<p align="center">
    <img src="./images/agg.png" />
</p>

More built-in aggregate functions to play:
```sql
/* Round up to a nearest integer */
SELECT ROUND(160.352323);

/* Round number up */
SELECT Ceil(16.3);

/* Round number down */
SELECT floor(16.8);

/* Get absolute value */
SELECT ABS(-6);

/* Get square root of a number */
SELECT SQRT(16);

/* Count number of characters in the string */
SELECT LENGTH('any_string');

/* convert to upper */
SELECT UPPER('any_string');

/* convert to lower case */
SELECT LOWER('any_STRING');

/* Get a substring */
SELECT SUBSTRING('any_string', 3)

/* replace a string */
SELECT REPLACE('main_string', 'to_be_replace_string', 'replace_string')
```

When we used `GROUP BY` above, we used it at the end, before `WHERE` clause. Then, how can we further do the filtering after having the grouped rows. We do this by using `HAVING` clause to do filter after group by objects have formed.
```sql
/* select query with HAVING constraint */
SELECT group_by_col, AGG_FUNC(col_expr) AS agg_result_a
FROM myTable
WHERE condition(s)
GROUP BY column
HAVING group_condition;
```
>The constraints/conditions that we can use after `HAVING` clause is similarly written as you would with `WHERE` clause.

## **Order of Execution** of a query & a Complete query:
Let's first see how all the parts of clause fit in, i.e. a complete picture:
```sql
/* complete SELECT query */
SELECT DISTINCT column, AGG_FUNC(column_or_expression), ???
FROM mytable
    JOIN another_table
      ON mytable.column = another_table.column
    WHERE constraint_expression
    GROUP BY column
    HAVING constraint_expression
    ORDER BY column ASC/DESC
    LIMIT count OFFSET COUNT;
```
Each part of the query is executed sequentially. See [sqlbolt](https://sqlbolt.com/lesson/select_queries_order_of_execution) for more details, but in short, the clause order of execution is as follows:
1. `FROM` and `JOIN` s: to determine the total working set of data
2. `WHERE`
3. `GROUP BY`
4. `HAVING`
5. `SELECT`
6. `DISTINCT`
7. `ORDER BY`
8. `LIMIT / OFFSET`

## How to **Create Tables** to put into the Database?
We can directly import `.csv` files into the database as a table with the data import wizard found in *mysql workbench*, and I think that will be the most efficient way to import table and put into database, then play your sql queries on them. Nonetheless, we can manually also add tables and so on. Let's see how. We can add tables into DB using `CREATE TABLE` statements.
```sql
/* create table */
/* default will let the col to take defualt value if not provided. */
CREATE TABLE IF NOT EXISTS myTable (
    col1 data_type table_constraint DEFAULT default_value,
    col2 data_type table_constraint DEFAULT default_value,
    ...
);

/* create table col with some constraints */
CREATE TABLE test ( a TEXT, b TEXT, c TEXT NOT NULL ); /* ensures col c is not NULL */
CREATE TABLE test ( a TEXT, b TEXT, c TEXT DEFAULT 'panda' ); /* takes default value if not provided later */
CREATE TABLE test ( a TEXT UNIQUE, b TEXT, c TEXT DEFAULT 'panda' ); /* UNIQUE don't allow duplicates */
CREATE TABLE test ( a TEXT UNIQUE NOT NULL, b TEXT, c TEXT DEFAULT 'panda' );

/* more examples */
/* PRIMARY KEY automatically creates id (starts from 1) for all rows */
/* a table can only have PRIMARY KEY col , works in sqlite3 system */
/* id col will populate itself, just add values for other col */
CREATE TABLE movies(
    id INTEGER PPRIMARY KEY,   /* PRIMARY KEY is a contraint to make values in col unique */
    title TEXT,
    director TEXT,
    year INTEGER,
    length_minutes INTEGER
);

/* create and fill table */
CREATE TABLE test (
  a INTEGER,
  b TEXT
);

INSERT INTO test VALUES ( 1, 'a' );
INSERT INTO test VALUES ( 2, 'b' );
INSERT INTO test VALUES ( 3, 'c' );
SELECT * FROM test;
```
Different DB supports different data types as a table column, but common ones are numeric types (double, float), string, booleans, dates, etc. See below image for more info:

<p align="center">
    <img src="./images/sql_dtypes.png" />
</p>

## Inserting, Updating, Deleting Rows in table:
```sql
INSERT INTO myTable
  VALUES (value _or_expr, another_value _or_expr, ...),
       (value _or_expr, another_value _or_expr, ...),
       ...;
       
/* Insert statement with specific columns */
INSERT INTO mytable (column, another_column, ???)
  VALUES (value_or_expr, another_value_or_expr, ???),
       (value_or_expr_2, another_value_or_expr_2, ???),
      ???;
      
/* Example Insert statement with expressions */
INSERT INTO boxoffice (movie_id, rating, sales_in_millions)
  VALUES (1, 9.9, 283742034 / 1000000);

INSERT INTO customer (name, address, city, state, zip) 
  VALUES ('Fred Flintstone', '123 Cobblestone Way', 'Bedrock', 'CA', '91234');
  
/* if we only insert on few columns of a row, then we get NULL on rest col */
INSERT INTO customer (name, city, state) 
  VALUES ('Jimi Hendrix', 'Renton', 'WA');
  
/* more insert into examples */
INSERT INTO test VALUES ( 1, 'This', 'Right here!' ); 
INSERT INTO test ( b, c ) VALUES ( 'That', 'Over there!' ); /* only inserting into certain cols */ 
INSERT INTO test DEFAULT VALUES; /* all the values for this row will be NULL */
INSERT INTO test ( a, b, c ) SELECT id, name, description from Item_table; /* using select of another table */

/* =============================== */
/* Update statement with values */
UPDATE mytable
  SET column = value_or_expr, 
    other_column = another_value_or_expr, 
    ???
  WHERE condition;
  
/* update table/row example */
UPDATE customer SET address = '123 Music Avenue', zip = '98056' WHERE id = 5; /* id col to specify row */
UPDATE customer SET address = '2603 S Washington St', zip = '98056' WHERE id = 5; /* change again */
UPDATE customer SET address = NULL, zip = NULL WHERE id = 5; /* finally make NULL back to it */

/* ================================ */
/* Delete statement with condition */
DELETE FROM mytable
  WHERE condition;
  
/* first select and see if what you see is what you want to delete */
SELECT * FROM customer WHERE id = 4;
DELETE FROM customer WHERE id = 4;

SELECT * FROM customer; /* to see that row is deleted */
```

![tip](./images/sql_tip.png)

## The `NULL` value:
`NULL` is special in that it lacks the value, it is a state where there is no value not even empty string, zero. Below we will see how `NULL` behaves in different situations:
```sql
/* first see the test table */
SELECT * FROM test;
/* because NULL is absence of value */
SELECT * FROM test WHERE a = NULL;    /* throws an error */
SELECT * FROM test WHERE a IS NULL;   /* this is how we can check for NULL */
SELECT * FROM test WHERE a IS NOT NULL;
INSERT INTO test ( a, b, c ) VALUES ( 0, NULL, '' ); /* this is okay, you can add NULL */
SELECT * FROM test WHERE b IS NULL;
SELECT * FROM test WHERE b = ''; /* error */
SELECT * FROM test WHERE c = '';
SELECT * FROM test WHERE c IS NULL; /* error */

DROP TABLE IF EXISTS test; /* delete above table */

/* create table with constraints that a,b col can't be NULL */
CREATE TABLE test (
  a INTEGER NOT NULL,
  b TEXT NOT NULL,
  c TEXT
);

INSERT INTO test VALUES ( 1, 'this', 'that' ); /* this passes */
SELECT * FROM test;

INSERT INTO test ( b, c ) VALUES ( 'one', 'two' ); /* error, a can't be NULL */
INSERT INTO test ( a, c ) VALUES ( 1, 'two' ); /* error, b can't be NULL */
INSERT INTO test ( a, b ) VALUES ( 1, 'two' ); /* this is fine, c can be NULL */
DROP TABLE IF EXISTS test; /* finally, deleting the test table */

/* To retrive all the rows with a missing value */
SELECT col1, col2 FROM myTable
WHERE col1 IS NULL;

/* to replace NULL with a value */
SELECT col1, COALESCE(col1, 'value_to_put')
FROM myTable;
```

## Altering tables:
With time, the table constraints might change, you need to update tables, or add new columns, or change data types of the columns. You can do those using `ALTER TABLE` statement.

```sql
/* adding column */
ALTER TABLE myTable
ADD col3 data_type;

/* removing column */
ALTER TABLE myTable
DROP col3;

/* renaming the table */
ALTER TABLE myTable
RENAME TO new_table_name;
```

## Delete / Dropping tables:
In rare cases, you might have to totally delete table from DB, you can use `DROP TABLE` statement.
```sql
DROP TABLE IF EXISTS myTable;

DROP TABLE test;
DROP TABLE IF EXISTS test;
```

## Finding length of string:
There is no standard implementation to find the length of the string, but here shown that works for sqlite.
```sql
SELECT LENGTH('string');

/* find length of name and order by length of name */
SELECT Name, LENGTH(Name) AS Len FROM City ORDER BY Len DESC;
```

## Selecting part of a String:
```sql
SELECT SUBSTR('this string', 6); /* starts from 6 pos, returns string */
SELECT SUBSTR('this string', 6, 3); /* returns str */
SELECT released, /* released data looks like this -> 1974-04-22 */
    SUBSTR(released, 1, 4) AS year, /* position 1 and 4 characters, and so on */
    SUBSTR(released, 6, 2) AS month,
    SUBSTR(released, 9, 2) AS day
  FROM album
  ORDER BY released
;
```

## Removing spaces from strings:
```sql
SELECT TRIM('   string   ');       /* removes spaces from both sides */
SELECT LTRIM('   string   ');      /* removes from left side */
SELECT RTRIM('   string   ');      /* removes from right side */
SELECT TRIM('...string...', '.');  /* removes specified character both sides */
```

## lower and upper case for strings:
```sql
SELECT 'StRiNg' = 'string'; /* not the same, so false */
SELECT LOWER('StRiNg') = LOWER('string'); /* now same */
SELECT UPPER('StRiNg') = UPPER('string'); /* true */
SELECT UPPER(Name) FROM City ORDER BY Name;
SELECT LOWER(Name) FROM City ORDER BY Name;
```

## Integer, real numeric types:
```sql
SELECT TYPEOF( 1 + 1 ); /* integer */
SELECT TYPEOF( 1 + 1.0 ); /* real */
SELECT TYPEOF('panda'); /* text */
SELECT TYPEOF('panda' + 'koala'); /* integer (in sqlite) */

SELECT 1 / 2; /* integer division, returns 0 */
SELECT 1.0 / 2;
SELECT CAST(1 AS REAL) / 2;
SELECT 17 / 5;
SELECT 17 / 5, 17 % 5; /* modulo gives remainder */

SELECT 2.55555;
SELECT ROUND(2.55555); /* round up to 3 */
SELECT ROUND(2.55555, 3); /* round up to 3 digit, 2.556 */
SELECT ROUND(2.55555, 0); /* still 3 */
```

## Dates and Times:
```sql
SELECT DATETIME('now');
SELECT DATE('now');
SELECT TIME('now');
SELECT DATETIME('now', '+1 day');
SELECT DATETIME('now', '+3 days');
SELECT DATETIME('now', '-1 month');
SELECT DATETIME('now', '+1 year');
SELECT DATETIME('now', '+3 hours', '+27 minutes', '-1 day', '+3 years');

/* to get the Year part from the DateTime object */
/* We can use YEAR() function that takes in DateTime and returns Year part from it */
SELECT firstName, lastName, Year(BirthDate) AS BirthYear
FROM myTable;
```

## Cheat-Sheet found from web:
List of some of the basic commands:

<p align="center">
    <img src="./images/sql_basic_commands.jpeg" />
</p>


# SQL Crash Course with Corise  -- by Sourabh Bajaj
Link [here](https://corise.com/course/sql-crash-course).

## Week 1
Let's see what we are expecting to learn in Week 1.

<p align="center">
    <img src="./images/wk1_overview.png" />
</p>

## Order of Precedence in SQL
1. Arithmetic operators
2. Concatenation
3. `=, <, >, <=, >= !=`
4. `IS NULL`
5. `LIKE`
6. `IN`
7. `BETWEEN`
8. `NOT`
9. `AND`
10. `OR`

### What is SQL?
SQL (Structured Query Language) is a programming language that is specifically designed to access and manipulate data stored within different databases. SQL allows you to pull data out of large databases to answer business questions. For example, at Amazon, you could ask questions like: "What are the top selling baby products in July-Sept?" At Netflix, you could ask: "How many people converted from trial to a full subscription?" And at CoRise, we can ask questions like "How many learners attended the first lecture for SQL Crash Course?"

### Why learn SQL?
* It can handle much larger volumes of data than Excel. You can handle *billions* of data points whereas Excel has a limit of about 1 million.
* You can start analyzing data and help your organization make better decisions without having to be dependent on Data Engineers and Data Scientists.
* Or you can be a software engineer that is trying to build cool and interesting apps.

Look at the following image of what is happening when we query on database.

<p align="center">
    <img src="./images/dbqsys.png" />
</p>

**Table:** A single table generally represents data about a single object. So at Airbnb, we would come across tables like: users, listings, bookings, reviews, etc. As a simple mental model, you should think of a table as a single spreadsheet in Excel.

**Database:** A database is a collection of tables that might be linked together via some defined relationships. So continuing our Airbnb example, we might have one database that contains all of our different tables.

**Query:** A query is a piece of SQL code that the database processes and takes action on. In the query that we ran in the previous section, it returned us with some data.


### Table Schemas - Columns and Rows
The structure of a table is described by its schema -?? the **schema** is the name of the table and two pieces of information about each column in the table:
* **Name** of the column
* **Data type** of the column (integer, text, date, etc.)

Data types refer to the set of valid values a column can take. There are many data types supported in SQL, but let's look at three for illustration purposes:
* **Integer** refers to whole numbers such as -23, 36, and 0
* **Text** refers to a string or an array of characters
* **Date** refers to a representation of a date, such as 1/1/2023

>**Excel vs Database:**
>NOTE: A single column in a table can only contain data of one type. So if a column is of type integer then we can only put numbers inside it. This isn't something that is enforced by spreadsheets. 

Tables in database are always related to each other in some ways. Here is an example where **listing** table is liked with *listing_id* is linked with *neighborhood_id* to **neighborhood** table.

<p align="center">
    <img src="./images/idlink.png" />
</p>

and neighborhood tablw will have info with neighborhood id and what neighborhood it is located in. In our learning we will use the following table schemas to easy learning the concepts.

<p align="center">
    <img src="./images/schemalin.png" />
</p>

**Comparing dates** To compare date-type object you can just treat them as text and put the yyyy-mm-dd form in single quotes. So a comparison would look something like;
```sql
SELECT * FROM reviews WHERE date < '2019-06-16';
```
















## Resources:

* [SQLBolt](https://sqlbolt.com/)
* [MySQL Docs](https://dev.mysql.com/doc/)
* [SQL Essential Training (LinkedIn)](https://www.linkedin.com/learning/sql-essential-training-3)
* [SQL Practice](https://www.sql-practice.com/)
* [TechTFQ Youtube](https://www.youtube.com/c/techTFQ)
* [DataInterview SQL Pad](https://sql.datainterview.com/)
* [DataLemur](https://datalemur.com/)
* [LearnSQL](https://learnsql.com/) -> for SQL course
* [StrataScratch](https://www.stratascratch.com/blog/categories/sql/) -> for practicing SQL queries
* [LeetCode](https://leetcode.com/problemset/database/) -> for practicing simple SQL queries
* [Mode SQL](https://mode.com/sql-tutorial/)
* [8weeks sql challenge](https://8weeksqlchallenge.com/)
* [sql zoo](https://sqlzoo.net/wiki/SQL_Tutorial)
