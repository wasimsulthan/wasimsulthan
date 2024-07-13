create database supermart_db;
use supermart_db;
create table cutomers(customer_id varchar(10) primary key,
customer_name varchar(30),
segment varchar(20),
age int,
country varchar(20),
city varchar(20),
state varchar(20),
postal_code int,
region varchar(10));
select * from cutomers;
create table products( product_id varchar(20) primary key,
category varchar(20),
subcategory varchar(20),
product_name varchar(50));
select * from products;
create table sales( order_line int,
order_id varchar(20) ,
order_date varchar(20),
ship_date varchar(20),
ship_mode varchar(20),
customer_id varchar(10) references cutomers(customer_id),
product_id varchar(20) references products(product_id),
sales float,
quantity int,
discount float,
profit float);
load data INFILE "D:/360digitmg/module 3 sql/assignment/7 operators/Datasets/Assignments_04_SQL_datasets/Sales.csv"
into table sales
fields terminated by ','
enclosed by '"'
lines terminated by '\n'
ignore 1 rows;	

select * from sales;
select distinct city from cutomers where region in ('north','east');
select * from sales where sales between 100 and 500;
select * from cutomers where customer_name like '% ____';
select * from sales order by discount > 0 desc;
select * from sales order by discount > 0 desc limit 10;
select sum(sales) from sales;
select count(*) as customercount from cutomers
group by (region='north') and (age>=20 and age<=30) ;
select avg(age) as average_age from cutomers group by region='east';
select max(age),min(age) from cutomers group by city='philadelphia';
select product_id,sum(sales) as totalsales_in_$ from sales group by product_id order by totalsales_in_$ desc;
select product_id,sum(quantity) as total_quantity from sales group by product_id;
select product_id,count(distinct order_id) as number_of_orders from sales group by product_id;
select product_id,max(sales) as maximum_sales from sales group by product_id; 
select product_id,min(sales) as minimum_sales from sales group by product_id; 
select product_id,avg(sales) as average_sales from sales group by product_id; 
