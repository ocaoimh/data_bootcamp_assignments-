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

SELECT c.name AS category, AVG(f.length) AS avg_running_time
FROM category c
JOIN film_category fc ON c.category_id = fc.category_id
JOIN film f ON fc.film_id = f.film_id
GROUP BY c.name
ORDER BY avg_running_time DESC;

```

3. What is the average running time of films by category?

```sql
SELECT f.title, COUNT(*) AS rental_count
FROM film f
JOIN inventory i ON f.film_id = i.film_id
JOIN rental r ON i.inventory_id = r.inventory_id
GROUP BY f.film_id
ORDER BY rental_count DESC;

```

5. Display the most frequently rented movies in descending order.

```sql
SELECT f.title, COUNT(*) AS rental_count
FROM film f
JOIN inventory i ON f.film_id = i.film_id
JOIN rental r ON i.inventory_id = r.inventory_id
GROUP BY f.film_id
ORDER BY rental_count DESC; 

```

6. List the top five genres in gross revenue in descending order.

```sql
SELECT c.name AS genre, SUM(p.amount) AS gross_revenue # creates a new column called genre
FROM category c
JOIN film_category fc ON c.category_id = fc.category_id
JOIN inventory i ON fc.film_id = i.film_id
JOIN rental r ON i.inventory_id = r.inventory_id
JOIN payment p ON r.rental_id = p.rental_id
GROUP BY c.name
ORDER BY gross_revenue DESC
LIMIT 5;


```

7. Is "Academy Dinosaur" available for rent from Store 1?

### I wasn't able to do this one by myself :( 

```sql
SELECT COUNT(*) AS available
FROM film f
JOIN inventory i ON f.film_id = i.film_id
WHERE f.title = 'Academy Dinosaur'
AND i.store_id = 1
AND NOT EXISTS (
    SELECT 1
    FROM rental r
    WHERE i.inventory_id = r.inventory_id
    AND (r.return_date IS NULL OR r.return_date > NOW())
);

```
