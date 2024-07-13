create database student_db;
use student_db;
create table  student_details(
id int primary key,
std_name varchar(25),
age int,
grade float);
insert into student_details values(1,"ram",12,80.0);
insert into student_details values(2,"kishore",14,86.0);
insert into student_details values(3,"hemanth",16,93.0);
insert into student_details values(4,"pooja",15,91.0);

create table student_details_copy(
id int primary key,
std_name varchar(25),
age int,
grade float);

DELIMITER //
create trigger after_insert_details after insert on student_details 
for each row 
begin 
insert into student_details_copy (id,std_name,age,grade) values (new.id,new.std_name,new.age,new.grade);
end;
//
delimiter ;
insert into student_details values(5,"rohith",14,76.0);
select * from student_details_copy;

DELIMITER //
create trigger update_grade before update on student_details
for each row 
begin 
if new.age<18 then
   set new.grade= new.grade*0.9;
elseif new.age between 18 and 20 then 
   set new.grade = new.grade*1.1;
else 
   set new.grade = new.grade * 1.05;
end if;
end;
//
delimiter ;

update student_details 	set age = 17 where id = 1;
select * from student_details;

#3 
# AFTER
# after trigger occurs after the event occurs like (update,insert,delete)
# it is used perform actions that should take place after changes to the table have been commited

# INSTEAD OF 
# instead of trigger occurs instead of the triggering event . it allows to replace the standard action of the triggering event with custom logic
# instead of triggers are useful when you solve complex business logic or handle specific cases that are not covered by the standard database operations.


#4
# The instead of delete trigger operator in SQL is used to replace the default action of a 
# delete operation on a view or table with custom logic defined in the trigger. 
# This type of trigger is particularly useful when you want to implement your own delete behavior for a 
# view or handle specific cases that are not covered by the standard delete operation.






		