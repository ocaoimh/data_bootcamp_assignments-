![logo_ironhack_blue 7](https://user-images.githubusercontent.com/23629340/40541063-a07a0a8a-601a-11e8-91b5-2f13e4e6b441.png)

# Lab | SQL Self and cross join

In this lab, you will be using the [Sakila](https://dev.mysql.com/doc/sakila/en/) database of movie rentals.

### Instructions

1. Get all pairs of actors that worked together.

```sql
SELECT a1.actor_id AS actor1_id, a1.first_name AS actor1_first_name, a1.last_name AS actor1_last_name,
       a2.actor_id AS actor2_id, a2.first_name AS actor2_first_name, a2.last_name AS actor2_last_name
FROM film_actor fa1
JOIN film_actor fa2 ON fa1.film_id = fa2.film_id AND fa1.actor_id < fa2.actor_id
JOIN actor a1 ON fa1.actor_id = a1.actor_id
JOIN actor a2 ON fa2.actor_id = a2.actor_id
GROUP BY actor1_id, actor2_id;

```

2. Get all pairs of customers that have rented the same film more than 3 times.

```sql
 SELECT customer.customer_id, customer.first_name, customer.last_name, film.title, COUNT(*) AS rental_count
FROM rental 
JOIN customer  ON rental.customer_id = customer.customer_id
JOIN inventory ON rental.inventory_id = inventory.inventory_id
JOIN film  ON inventory.film_id = film.film_id
GROUP BY customer.customer_id, film.film_id
HAVING rental_count > 2
ORDER BY customer.customer_id, rental_count DESC;

```

3. Get all possible pairs of actors and films.


```sql
SELECT a.actor_id, a.first_name, a.last_name, f.film_id, f.title
FROM actor a
JOIN film_actor fa ON a.actor_id = fa.actor_id
JOIN film f ON fa.film_id = f.film_id
ORDER BY a.actor_id, f.film_id;

```


```sql
# Extra example 
#Get all possible pairs of actors and film categories
# 3 tables actor, film_category, & film

SELECT actor.actor_id, actor.first_name, actor.last_name, category.name
FROM actor, film_actor, film_category, category
WHERE actor.actor_id = film_actor.actor_id
AND film_actor.film_id = film_category.film_id
AND film_category.category_id = category.category_id
ORDER BY actor.actor_id, category.name;
```