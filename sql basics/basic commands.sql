create database classroom;
use classroom;
create table science_class(
enrollment_no int,
stdname varchar(20),
science_marks int);
insert into science_class values(1,"popeye",33);
insert into science_class values(2,"olive",54);
insert into science_class values(3,"brutus",98);
select * from science_class;
select stdname from science_class where science_marks > 60;
select * from science_class where (science_marks > 35 and science_marks < 60); 
select * from science_class where (science_marks<= 35 or science_marks>=60);
update science_class set science_marks = 45 where stdname = "popeye" ;
delete from science_class where stdname ="robb";
alter table science_class rename column stdname to student_name;
select * from science_class;

