create database store;
use store;
create table sales(
order_id int,
customer_name varchar(20),
product char (15),
order_date date,
order_time time,
time_stamp timestamp,
return_status boolean,
net_amount decimal(10,2),
feedback text);
insert into sales values (1,'praveen','soap','2023-11-08','3:33:00',now(),0,30.00,'comment 1');
insert into sales values (2,'aadhitya','biscut','2023-11-08','3:33:00',now(),0,60.00,'comment 2');
insert into sales values (3,'suresh','bread','2023-11-08','3:33:00',now(),1,30.00,'comment 3');
insert into sales values (4,'lavanya','coffee powder','2023-11-08','3:33:00',now(),0,160.00,'comment 4');
insert into sales values (5,'kamal','chocolate','2023-11-08','3:33:00',now(),0,100.00,'comment 5');
insert into sales values (6,'sangeetha','soft drinks','2023-11-08','3:33:00',now(),0,40.00,'comment 6');
insert into sales values (7,'naveen','chips','2023-11-08','3:33:00',now(),1,20.00,'comment 7');
insert into sales values (8,'ramesh','soda','2023-11-08','3:33:00',now(),0,50.00,'comment 8');
insert into sales values (9,'yunus','stationaries','2023-11-08','3:33:00',now(),0,100.00,'comment 9');
insert into sales values (10,'sedhu','sweets','2023-11-08','3:33:00',now(),0,200.00,'comment 10');

alter table sales modify net_amount float;
alter table sales modify feedback varchar(40);
select * from sales;


#5. BLOB: is a binary large object data type which is used to store binary datas such as audio, video and other non-text data in database 
# There are 4 types of BLOB data type:
# BLOB: this normal binary large object
# TINYBLOB: stores small blob vlues
# MEDIUMBLOB: stores medium sized BLOB
# LARGE : stores large BLOB

#6. character datatypes:
# Char (n) : fixed length character with specified length
# Varchar(n): variable length character string
# Text(): variable length character string for storing large text data

# Numerical datatypes:
# Int(): integer datatype for whole number
# Float(): floating point number with a specified precision
# Decimal(n,m): fixed point number with n total digits and m decimal places

