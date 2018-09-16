# 第一章 背景知识
主键：用于唯一标识表中每个行的一个或多个列

外键：一个或多个用于识别**其他表**中某一行的列

# 第二章 创建和使用数据库
## 2.1 创建MySQL数据库
``` SQL
mysql -u root -p /*用root登录*/
create database bank;
grant all on bank.* to 'lrngsql'@'localhost' identified by 'xyz';
quit;

mysql -u lrngsql -p;
use bank; /*关联数据库*/
```

使用mysql命令行工具,同时指定用户名和要使用的数据库
``` SQL
mysql -u lrngsql -p bank
```

## 2.3 MySQL 数据类型
### 字符型数据
定长 char(20)

变长 varchar(20)

指定字符集 Varchar(20) character set gb2312\
指定整个数据库的默认字符集create database foreign_sales character set utf8

文本数据tinytext, text, mediumtext, longtext

### 数值型数据
整数类型tinyint, smallint, mediumint, int, bigint

浮点类型float(p,s),double(p,s)\
p数字总位数，s小数点后数字位数

### 时间数据
时间类型date, datetime, timestamp, year, time

## 2.4 表的创建
### 第一步：设计
### 第二步：精化
考虑数据类型，没有重复。

选择主键

子表及其自己的主键，确定指向主表的外键

### 第三步：构建SQL语句
```SQL
CREATE TABLE person
(person_id SMALLINT unsigned,
fname VARCHAR(20),
lname varchar(20),
gender CHAR(1),  /*可以把检查约束和数据类型定义在一起,
gender ENUM('M','F')*/
birth_date DATE,
street VARCHAR(20),
city varchar(20),
state VARCHAR(20),
country varchar(20),
postal_code varchar(20),
CONSTRAINT pk_person PRIMARY KEY (person_id)
); /*主键约束，创建在person_id上，且被命名为pk_person*/
```
```SQL
desc person; /*检查定义*/
```
```SQL
create table favorite_food
(person_id SMALLINT unsigned,
food varchar(20),
constraint pk_favorite_food primary key (person_id, food),
constraint fk_fav_food_person_id foreign key (person_id)
references person (person_id)
);/*外键约束，person_id列只能来自perosn表*/
```
## 2.5操作与修改表
### 插入数据
为主键打开自增特性
```SQL
SET FOREIGN_KEY_CHECKS = 0;
ALTER TABLE person MODIFY person_id SMALLINT UNSIGNED auto_increment;
SET FOREIGN_KEY_CHECKS = 1;
```


使用INSERT插入：\
三部分 表名，列名，列值
```SQL
INSERT INTO person
(person_id, fname, lname, gender, birth_date)
values (null, 'William', 'Turner', 'M', '1972-05-27');
```

使用SELECT 和 WHERE查找\
三部分：select列，from表，where条件
```SQL
SELECT person_id , fname, lname, birth_date
FROM person
where person_id = 1;

SELECT person_id, fname, lname, birth_date
from person
where lname = 'Turner';
```
### 更新数据
使用update更新数据\
三部分：update表，set 列=值，where条件
```SQL
update person
set street = '1225 tremont st.',
city = 'Boston',
state = 'MA',
country = 'USA',
postal_code = '01238'
where person_id = 1;
```

### 删除数据
使用delete删除数据\
delete from表，where条件
```SQL
Delete from person
where person_id = 2;
```

### 删除表
```SQL
drop  table person;
```

###查看可用的表
```
show tables;
```
## 2.6 导致错误的语句
### 主键不唯一
插入的主键位置已经有了，会出现错误。不如NULL，发挥自增性质
### 不存在的外键
reference person (person_id)约束了person_id列的值必须来自person表
### 无效的日期转换
```SQL
update person
/*set birth_date = 'DEC-21-1980'不合法*/
set birth_date = str_to_date ('DEC-21-1980', '%b-%d-%Y')
/*str_to_date是字符串转为日期的函数*/
where person_id = 1;
```

# 第三章 查询入门
## 3.3 select子句
**select子句用于在所有可能的列中，选择查询结果集要包含哪些列**

查询全部列
```SQL 
select *
from department;
```

select子句可以带上字符、表达式、内建函数、用户自定义函数
```SQL
select emp_id,
'active' as status,
emp_id * 3.14 as empid_x_pi,
upper(lname) last_name_upper
from employee;
```

```Sql
select version(),
user(),
database();
```

列的别名，as也可以省略
```sql
selece emp_id*3.14 as empid_x_pi
```

去除重复的行
```sql
select distinct cust_id
from account;
```

## 3.4 from子句
三种类型的表
1. 永久表。create table语句创建的表
2. 临时表。子查询select返回的表
3. 虚拟表。使用create view子句创建的视图

子查询产生的表
```sql
select e.emp_id, e.fname
from (select emp_id, fname, lname, title
from employee) e;
```

视图,视图被创建后，没有产生或储存任何数据。服务器只是保留该查询以供使用
```sql
create view employee_vw AS
select emp_id, fname, lname, year(start_date) start_year
from employee;

select emp_id, start_year
from employee_vw;
```

表连接,定义表别名
```sql
select e.emp_id, e.fname, e.lname, d.name dept_name
from employee as e inner join department as d
on e.dept_id = d.dept_id;
```

where子句，and和or
```sql
select account_id, cust_id, status, avail_balance 
from account 
where (status = 'ACTIVE') and (avail_balance > 2500);
```

group by 和having子句\
根据列值对数据进行分组。having子句搭配group by子句，功能类似where对分组数据进行过滤
```sql
select d.name, count(e.emp_id) as num_employees
from department d inner join employee as e
on d.dept_id = e.dept_id
group by d.name
having count(e.emp_id)>2;
```

oder by子句，升序或降序

根据表达式排序。right提取右数三个字符
```
order by right(fed_id,3) desc;
```

# 第四章 过滤
**where 语句限制了所选择的行数**

## 4.1 条件过滤
AND OR NOT

## 4.2 构建条件
表达式可以由下面类型中的任意一个：
1. 数字
2. 表或视图中的列
3. 字符串
4. 内建函数
5. 子查询
6. 表达式列表
7. 比较算符= != < > <> like in between
8. 算术操作符 + - * /

## 4.3 条件类型
相等条件
```sql
 where product_type_cd = 'ACCOUNT'
 ```
 不等条件
<>或!=

使用相等条件修改数据
```sql
set sql_safe_updates = 0;
delete from account
where status = 'CLOSED' and YEAR(close_date) = 2002;
```

范围条件 值是否处于某个区间 < > between
```SQL
where start_date < '2007-01-01';

where start_date between '2005-01-01' and '2007-01-01';/*必须左小右大*/
```
 
成员条件 IN 以及NOT IN
```sql
where product_cd in (select product_cd from product
where product_type_cd = 'ACCOUNT');
 ```
 
匹配条件
1. 使用通配符\
_正好1个字符\
%任意数目的字符
```sql
 select lname
 from employee
 /*where fed_id like '___-__-____';*/
 where lname like '%ba%';
```
2. 正则表达式

## 4.4 NULL
只能IS NULL 不能 = NULL

IS NOT NULL

注意确定哪些列可以允许NULL值
一些情况如允许null值，通过!= 比较算符会导致null值被忽略

# 第五章 多表查询
## 5.1 连接

```sql
select e.emp_id, e.fname, e.lname, b.name
from employee e inner join branch b
on e.assigned_branch_id = b.branch_id;
```
on的作用机理：e中的值1去找b中branch_id列为1的行，获取该行的name
值

## 5.2 连接三个或更多表
顺序交换，得到的结果是一致的
```sql
select c.cust_type_cd, c.fed_id, p.name 
from customer c inner join account a
on c.cust_id = a.cust_id
inner join product p
on a.product_cd = p.product_cd
where c.cust_type_cd = 'B';
```

连续两次使用同一个表。选取出两个列，一个是开户行，一个是开户柜员所属行。
branch是行代号和行名称的表，被用到两次。需要两个branch表的实例别名取不同，方便区分。
```sql
 select a.account_id, e.emp_id, b_a.name as open_branch, b_a.name emp_branch
from account a inner join branch b_a
on a.open_branch_id = b_a.branch_id
inner join employee e
on a.open_emp_id = e.emp_id
inner join branch b_e
on e.assigned_branch_id = b_e.branch_id
where a.product_cd = 'CHK'
order by emp_id;
```

自连接。因为表中包含了一个指向自身的外键，即指向本表主键的列。如列出雇员主管的名称。
```sql
select e.fname, e.lname, e_mgr.fname as mgr_fname, e_mgr.lname as mgr_lname
from employee e inner join employee e_mgr
on e.superior_emp_id = e_mgr.emp_id;
```
## 5.4 相等连接和不等连接
不与自身相同。如柜员之间安排下棋，不能与自己下棋，且每两个人不能重复匹配
```sql
select e1.fname, e1.lname, 'VS' vs, e2.fname, e2.lname
from employee e1 inner join employee e2
on e1.emp_id < e2.emp_id
where e1.title = 'Teller' and e2.title = 'Teller';
```

# 第六章 使用集合
## 6.1 集合理论基础
1. union并集
2. intersect交集
3. except（A except B 集合A减去集合B的结果）

两个数据集执行集合操作时，必须满足两个规范：
1. 两个数据集有同样数目的列
2. 两个数据集中对应列的数据类型必须是一样的

select语句中可以使用集合操作符执行集合操作，**每个select查询语句都会产生一个包含单个行的数据集合**

## 6.3 集合操作符
union和union all操作符\
union删除重复项
```sql
select 'ind' type_cd, cust_id, lname
from individual
union all
select 'bus' type_cd, cust_id, name
from business
union
select 'bus' type_cd, cust_id, name
from business
order by lname;
```
intersect和except操作符MySQL暂不支持

order by指定要排序的列时需要从复和查询(select)的第一个查询中选择列名

# 第七章 数据生产、转换和操作
## 7.1 使用字符串数据
字符数据类型
1. char 固定长度
2. varchar 边长字符串
3. text 容纳大长度的变长字符串

```sql
create table string_tbl
(char_fld char(30),
vchar_fld varchar(30),
text_fld text);
```


### 7.1.1 生成字符串
```sql
insert into string_tbl (char_fld, vchar_fld, text_fld)
values ('this is char data', 
'this is varchar data',
'this is text data');

update string_tbl
set vchar_fld = 'this is a piece of extremely long varchar data';
```

包含单引号
需要在单引号前加'作为转义符
```sql
update string_tbl
set text_fld = 'this string didn\'t work. but it does now.';
```

quote() 使用单引号将字符串包含起来，并为字符串中单引号添加转义符

concat() 连接\
特殊符号
```sql
select concat('danke sch', char(148), 'n');
```

### 7.1.1操作字符串
#### 返回数字的字符串函数

length() 字返回字符串长度

position() 返回子字符串位置\
locate() 返回字符串位置，可以指定搜索的起点\
注意第一个字符位置号为1
```sql
select position('is' in vchar_fld)
from string_tbl；

select locate('is',vchar_fld, 4)
from string_tbl;
```

Strcmp() 比较字符串前后位置。-1表示第一个字符串在第二个字符串之前，0表示两个字符串相同，1表示第一个字符串在第二个字符串之后
```sql
insert into string_tbl (vchar_fld) values('abcd');
insert into string_tbl (vchar_fld) values('xyz');
select strcmp('xyz', 'abcd') xyz_abcd;
```


like 操作符比较是否出现特定字符串
```
select name, name like '%ns' ends_in_ns
from department;
```

正则表达式

#### 返回字符串的字符串函数
concat()
```sql
select concat(fname,' ', lname, 'has been a ', title, 'since ', start_date) emp_narrative 
from employee
where title = 'Teller' or title = 'Head Teller';
```

insert() 在字符串中间增加或替换部分字符。insert接收四个参数，原始字符串，字符串操作开始的位置，需要替换的字符串，替换入的字符串。如果第三个参数为0，表示向右排放
```sql
update string_tbl
set vchar_fld = insert(vchar_fld, 1, 0,'insert sth');
```

substring() 提取字符串
```
select substring(vchar_fld, 2,3)  /*第2个位置提取3个字符*/
from string_tbl;
```

## 7.2 使用数值数据
### 7.2.1 执行算术运算
+、-、*、/

执行算术运算，各种Acos(), Asin(), Atan()函数

mod() 求余数
```
select mod(10,2);
```

pow() 求幂
```
select pow(2,3) '2^8';
```

### 7.2.2 控制数字精度
ceil向上取整， floor向下取整
```sql
select ceil(72.44445),floor(72.2131);
```

round()
1. 四舍五入
2. 四舍五入保留指定位数。负数取整
```
select round(72.3333,2);
```

truncate() 小数点后需要被截取多少位。若为负数，表示取整多少位-1代表10


### 7.2.3 处理有符号数
sign()返回1代表正数

abs()

## 7.3 使用时间数据

时区

### 7.3.2 生成时间数据
1. 从已有的date、datetime或time列中复制数据
2. 执行返回date、datetime或time型数据的内建函数
3. 构建可以被服务器识别的代表日期的字符串

#### 表示日期数据的字符串
- DATE YYYY-MM-DD
- DATETIME YYYY-MM-DD HH:MI:SS
- TIMESTAMP YYYY-MM-DD HH:MI:SS
- TIME HHH:MI:SS

服务器可以接受datetime类型字符串，将对其自动转换
```sql
update transaction
set txn_date = '2008-09-17 15:30:30'
where txn_id = 1;
```

#### 字符串到日期的转换
还有一种手动转换为datetime的方法，cast()
```sql
select cast('2008-09-01' as date) date_field;
```

#### 产生日期的函数
根据字符串产生时间数据，但不是标准格式cast接受不了，使用内建函数str_to_date()转化为日期字符串。各种符号意义见书
```
select str_to_date('01 17, 2008','%m %d, %Y') set_new_date;
```

当前日期
current_date()
current_time()
current_timestamp()

### 7.3.3 操作时间数据
#### 返回日期的时间函数

加上一段时间date_add(current_date(),interval 5 day)

当月最后一天last_dat()

转换时区convert_tz()

#### 返回字符串的时间函数
提取日期信息select extract(year from '2008-01-01');

确定星期几dayname('2008-01-01');

#### 返回数字的时间函数
时间间隔 select datediff('2018-09-01','2018-01-01');

## 7.4 转换函数
cast() 将**字符串**转换为指定数据类型
```
select cast('-111' as signed integer);
```


# 第八章

## 8.2 聚集函数
- max()
- min
- avg
- sum
- count

```
select product_cd,
max(avail_balance) max_balance,
min(avail_balance) min_balance,
avg(avail_balance) avg_balance,
sum(avail_balance) tot_balance,
count(*) numaccounts
from account
group by product_cd;
```

### 8.2.2 对独立值计数
```
select count(distinct open_emp_id)
from account;
```

### 8.2.3 使用表达式
```sql
select max(pending_balance - avail_balance) max_uncleared
from account;
```

### 8.2.4 null值
count(*)对所有行计数\
count(列名)忽略null值

## 8.3 产生分组
### 8.3.1 对单列分组
```
select product_cd, sum(avail_balance) prod_balance
from account
group by product_cd;
```

### 8.3.2 对多列分组
```
select product_cd, open_branch_id,
sum(avail_balance) prod_balance
from account
group by product_cd, open_branch_id
order by product_cd;
```
### 8.3.2 利用表达式分组
```
select extract(year from start_date) year,
count(*) how_many
from employee
group by extract(year from start_date);
```

### 8.3.4 产生合计数
分别求和再对product_cd单独求和，求出所有行总和
```
select product_cd, open_branch_id,
sum(avail_balance) tot_balance
from account
group by product_cd, open_branch_id with rollup;
```

## 8.4 分组过滤条件
where应该在分组之前执行，having过滤条件在分组之后搭配使用
```
select open_emp_id, count(*) how_many
from account
where status = 'ACTIVE'
group by open_emp_id
having count(*) > 4;
```

# 第九章 子查询
总结：
1. 它返回的结果可以是单列/单行，单列/多行，多列/多行，如果返回结果多于1行，他可以用于比较，而不能用于等式判断
2. 它可以独立于包含语句（非关联子查询）
3. 它可以引用包含语句中一行或多行（关联子查询）
4. 它可以用于where条件中，这些where条件使用比较算符以及其他特殊目的的算符（in, not in, exists, not exists）；
5. 它可以出现于select, update, delete, insert
6. 它产生的结果集可以与其他表或者子查询连接
7. 它可以生成值用来insert填充表或者一些列
8. 可以用于select、from、where、having、order by

## 9.3 非关联子查询
单行单列子查询结果，成为标量子查询。可以位于= <> < > <= >= 任意一边
```sql
select account_id
from account
where open_emp_id <> (select e.emp_id
from employee e inner join branch b
on e.assigned_branch_id = b.branch_id
where e.title = 'Head Teller' and b.city = 'Woburn');
```

### 9.3.1 多行单列子查询
#### in和not in
```
select emp_id
from employee
where emp_id not in (null);
```

使用not in或<> 比较一个值和一个值集的，读者必须注意确保值集中不含null值。否则会出现错。null与is not搭配

#### all运算符
与<> < > =搭配使用

#### any运算符
只要有一个比较成立，则条件为真

### 9.3.2 多列子查询
```sql
select account_id
from account
where open_branch_id = (select branch_id
from branch
where name = 'Woburn Branch')
and open_emp_id in (select emp_id
from employee
where title = 'Teller' or title = 'Head Teller');
/*使用了两个单列子查询*/
```

## 9.4 关联子查询
先执行包含查询，再执行子查询

```
select c.cust_id
from customer c
where 2 = (select count(*)
from account a
where a.cust_id = c.cust_id);
/*先从customer表中检索出13行，接着为每一行执行子查询，
每次执行都要包含查询向子查询传递传递客户ID
若子查询返回值2，则过滤调价满足，该行才被添加到结果集*/
```

#### 9.4.1 exists运算符
若只关心关系而不在于数量。查询是否至少返回1行
```
select a.account_id,  a.product_cd, a.cust_id, a.avail_balance
from account a
where exists (select 1
from transaction t
where t.account_id = a.account_id
and t.txn_date = '2003-07-30');
```

### 9.4.2 关联子查询update数据
```
set sql_safe_updates = 0;
update account a
set a.last_activity_date =
(select max(t.txn_date)
from transaction t
where t.account_id = a.account_id)
where exists (select 1
from transaction t
where t.account_id = a.account_id);
/*exist 查询每个账户是否发生过交易*/
```

## 9.5 何时使用子查询
### 9.5.1 子查询做为数据源
子查询在from子句中使用必须是非关联的，它首先执行然后一直保留于内存中直至包含查询执行完毕。
```sql
select d.dept_id
from department d inner join
(select dept_id, count(*) how_many
from employee
group by dept_id) e_cnt
on d.dept_id = e_cnt.dept_id;
```

#### 数据加工
如对客户按照账户里余额多少进行分组

#### 面向任务的子查询
以下product、branch、employee这三个表只是用于描述。account表已有分组所需的信息（product_cd, open_branch_id, open_emp_id, avail_balance） 将分组任务独立出来，然后将子查询生成的三个表连接成一个表，最后得到最终结果。
```sql
select p.name product, b.name branch,
concat(e.fname, ' ', e.lname) name,
account_groups.tot_deposits
from
(select product_cd, open_branch_id branch_id,
open_emp_id emp_id,
sum(avail_balance) tot_deposits
from account
group by product_cd, open_branch_id, open_emp_id) account_groups
inner join employee e on e.emp_id = account_groups.emp_id
inner join branch b on b.branch_id = account_groups.branch_id
inner join product p on p.product_cd = account_groups.product_cd
where p.product_type_cd = 'ACCOUNT';
```

### 9.5.2 过滤条件中的子查询
在having中使用子查询来查找开户最多的雇员

### 9.5.3 子查询作为表达式生成器
用于select, order by, insert中的values语句

# 第十章 再谈连接
## 10.1 外连接
left外连接包括第一个表的所有行，但仅仅包含第二个表中那些匹配行的数据。right外连接则相反
```sql
select a.account_id, a.cust_id, b.name
from account a left outer join business b
on a.cust_id = b.cust_id;
```

### 10.1.2 三路外连接
```
select a.account_id, a.product_cd,
concat(i.fname, ' ', i.lname) person_name,
b.name business_name
from account a left outer join individual i
on a.cust_id = i.cust_id
left outer join business b
on a.cust_id = b.cust_id;
```

### 10.1.3 自外连接
将employee表连接到自己而生成雇员和他们主管的列表
```
select e.fname, e.lname,
e_mgr.fname mgr_fname, e_mgr.lname mgr_lname
from employee e left outer join employee e_mgr
on e.superior_emp_id = e_mgr.emp_id; /*!!!*/
```

## 10.2 交叉连接
笛卡尔积连接
```
select days.dt, count(t.txn_date) /*与group by搭配使用*/
from transaction t right outer join
(select date_add('2008-01-01',
interval (ones.num + tens.num + hundreds.num) day) dt 
from 
(select 0 num union all
select 1 num union all
select 2 num union all
select 3 num union all
select 4 num union all
select 5 num union all
select 6 num union all
select 7 num union all
select 8 num union all
select 9 num) ones
cross join
(select 0 num union all
select 10 num union all
select 20 num union all
select 30 num union all
select 40 num union all
select 50 num union all
select 60 num union all
select 70 num union all
select 80 num union all
select 90 num) tens
cross join
(select 0 num union all
select 100 num union all
select 200 num union all
select 300 num) hundreds
where date_add('2008-01-01', interval (ones.num + tens.num + hundreds.num) day) <
'2009-01-01') days
on days.dt = t.txn_date 
group by days.dt
order by 1; /*按第一列排序*/
```

## 10.3 自然连接
自然连接 是指依赖多表交叉时的相同列名来自动推断正确的连接条件。如果没有相同的列名，不要这么做会出现错误
```
select a.account_id, a.cust_id
from account a natural join customer c;
```

# 第十一章 条件逻辑
## 11.2 case表达式
### 11.2.1 查找型case表达式
```sql
select c.cust_id, c.fed_id,
  case
    when c.cust_type_cd = 'I'
      then concat(i.fname, ' ', i.lname)
	when c.cust_type_cd = 'B'
      then b.name
	else 'Unknow'
  end name
from customer c left outer join individual i
  on c.cust_id = i.cust_id
  left outer join business b
  on c.cust_id = b.cust_id;
  ```
  

语法如下 
```sql
 case (c0)
  when c1 then e1
  when c2 then e2
  (else ed)
end
```

### 11.2.2 简单case表达式
语法如下，V0代表一个值，符号V1,V2,...VN代表要与V0比较的值
```
case V0
  when V1 then e1
  when V2 then e2
  (else ed)
end
```

count计算一个值再进行比较
```
select c.cust_id, c.fed_id, c.cust_type_cd,
  case (select count(*) from account a
    where a.cust_id = c.cust_id)
    when 0 then 'None'
    when 1 then '1'
    when 2 then '2'
    else '3+'
  end num_accounts
from customer c;
```

## 11.3 case表达式范例
### 11.3.1 结果集变换
### 11.3.2 选择性聚合
### 11.3.3 存在性检查
与exist搭配使用
```
select c.cust_id, c.fed_id, c.cust_type_cd,
  case
    when exists (select 1 from account a
      where a.cust_id = c.cust_id
      and a.product_cd = 'CHK') then 'Y'
	else 'N'
  end has_checking,
  case
    when exists (select 1 from account a
	  where a.cust_id = c.cust_id
      and a.product_cd = 'SAV') then 'Y'
	else 'N'
  end has_savings
from customer c;
```
### 11.3.4 除零错误
```
select a.cust_id, a.product_cd, a.avail_balance/
  case
    when prod_tots.tot_balance = 0 then 1
    else prod_tots.tot_balance
  end percent_of_total
from account a inner join
(select a.product_cd, sum(a.avail_balance) tot_balance
from account a
group by a.product_cd) prod_tots
on a.product_cd = prod_tots.product_cd;
```

### 11.3.6 null值处理
```
select <some calculation>
  case 
    when avail_balance is null then 0
    else avail_balance
  end
```