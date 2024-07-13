create table sales_join as select * from supermart_db.sales;
select count(distinct customer_id) as distinct_customer from sales_join;
create table customer_20_60 as select * from supermart_db.cutomers where age between 20 and 60;
select count(*) from customer_20_60;
select customer_20_60.state,sum(sales_join.sales) as totalsales from customer_20_60  join sales_join on 
customer_20_60.customer_id = sales_join.customer_id group by customer_20_60.state;
select products.product_id,products.product_name,products.category,sum(sales_join.sales) as total_sales,
sum(sales_join.quantity) as total_quantity from products join sales_join on products.product_id = sales_join.product_id
group by products.product_id,products.product_name,products.category;
