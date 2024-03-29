![logo_ironhack_blue 7](https://user-images.githubusercontent.com/23629340/40541063-a07a0a8a-601a-11e8-91b5-2f13e4e6b441.png)

# Lab | Stored procedures

In this lab, we will continue working on the [Sakila](https://dev.mysql.com/doc/sakila/en/) database of movie rentals. 

### Instructions

Write queries, stored procedures to answer the following questions:

- In the previous lab we wrote a query to find first name, last name, and emails of all the customers who rented `Action` movies. Convert the query into a simple stored procedure. Use the following query:

  ```sql
    select first_name, last_name, email
    from customer
    join rental on customer.customer_id = rental.customer_id
    join inventory on rental.inventory_id = inventory.inventory_id
    join film on film.film_id = inventory.film_id
    join film_category on film_category.film_id = film.film_id
    join category on category.category_id = film_category.category_id
    where category.name = "Action"
    group by first_name, last_name, email;
  ```

- Now keep working on the previous stored procedure to make it more dynamic. Update the stored procedure in a such manner that it can take a string argument for the category name and return the results for all customers that rented movie of that category/genre. For eg., it could be `action`, `animation`, `children`, `classics`, etc.

- Write a query to check the number of movies released in each movie category. Convert the query in to a stored procedure to filter only those categories that have movies released greater than a certain number. Pass that number as an argument in the stored procedure.


### Solution
Here's the proposed solution 


```sql
use sakila;

-- the initial query
  
  select first_name, last_name, email
  from customer
  join rental on customer.customer_id = rental.customer_id
  join inventory on rental.inventory_id = inventory.inventory_id
  join film on film.film_id = inventory.film_id
  join film_category on film_category.film_id = film.film_id
  join category on category.category_id = film_category.category_id
  where category.name = "Action"
  group by first_name, last_name, email;
  
  
  
    
  -- make the query a stored procedure
DELIMITER //
CREATE PROCEDURE GetCustomersByCategoryAction()
BEGIN
    SELECT first_name, last_name, email
    FROM customer
    JOIN rental ON customer.customer_id = rental.customer_id
    JOIN inventory ON rental.inventory_id = inventory.inventory_id
    JOIN film ON film.film_id = inventory.film_id
    JOIN film_category ON film_category.film_id = film.film_id
    JOIN category ON category.category_id = film_category.category_id
    WHERE category.name = "Action"
    GROUP BY first_name, last_name, email;
END //
DELIMITER ;

  
  -- Now keep working on the previous stored procedure to make it more dynamic. Update the stored procedure in a such manner that it can take a string argument for the category name and return the results for all customers that rented movie of that category/genre. For eg., it could be action, animation, children, classics, etc.
  


DELIMITER //
CREATE PROCEDURE GetCustomersByCategoryDynamic(IN categoryName VARCHAR(255))
BEGIN
    SET @query = CONCAT('
        SELECT first_name, last_name, email
        FROM customer
        JOIN rental ON customer.customer_id = rental.customer_id
        JOIN inventory ON rental.inventory_id = inventory.inventory_id
        JOIN film ON film.film_id = inventory.film_id
        JOIN film_category ON film_category.film_id = film.film_id
        JOIN category ON category.category_id = film_category.category_id
        WHERE category.name = "', categoryName, '"
        GROUP BY first_name, last_name, email;'
    );
    PREPARE stmt FROM @query;
    EXECUTE stmt;
    DEALLOCATE PREPARE stmt;
END //
DELIMITER ;

CALL GetCustomersByCategoryDynamic('Animation');

-- write a query to see the nb of movies released per catgory
SELECT category.name, COUNT(film.film_id) AS movie_count
FROM category
JOIN film_category ON category.category_id = film_category.category_id
JOIN film ON film_category.film_id = film.film_id
GROUP BY category.name;


-- convert the query to a stored procedure to filter only those movies that are greater than a certain number. 

DELIMITER //
CREATE PROCEDURE FilterCategoriesByMovieCount(IN minMovieCount INT)
BEGIN
    SELECT category.name, COUNT(film.film_id) AS movie_count
    FROM category
    JOIN film_category ON category.category_id = film_category.category_id
    JOIN film ON film_category.film_id = film.film_id
    GROUP BY category.name
    HAVING movie_count > minMovieCount;
END //
DELIMITER ;

-- user input to select minimum number of movies 
CALL FilterCategoriesByMovieCount(30);
```
