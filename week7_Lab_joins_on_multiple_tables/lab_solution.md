![logo_ironhack_blue 7](https://user-images.githubusercontent.com/23629340/40541063-a07a0a8a-601a-11e8-91b5-2f13e4e6b441.png)

# Lab | SQL Joins on multiple tables

In this lab, you will be using the [Sakila](https://dev.mysql.com/doc/sakila/en/) database of movie rentals.

### Instructions

1. Write a query to display for each store its store ID, city, and country.

```sql
SELECT s.store_id, c.city, co.country # these are the three columns we'll make
FROM store s # this has address id
JOIN address a # this has address id
ON s.address_id = a.address_id # create a join on these 2 columns 
JOIN city c # this table has address_id and city_id
ON a.city_id = c.city_id #join the address and store cols to the city col 
JOIN country co # this has country and city ids
ON c.country_id = co.country_id; 
```


2. Write a query to display how much business, in dollars, each store brought in.

```sql
 

```

3. What is the average running time of films by category?
```sql
 

```

4. Which film categories are longest?

```sql
 

```

5. Display the most frequently rented movies in descending order.

```sql
 

```

6. List the top five genres in gross revenue in descending order.

```sql
 

```

7. Is "Academy Dinosaur" available for rent from Store 1?

```sql
 

```
