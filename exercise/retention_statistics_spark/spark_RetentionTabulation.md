# Spark exercise - Handling a large retail data
This workflow assumes that you have installed Apache Spark and pyspark successfully, along with internet access to acquire data used.

The data is pulled from [University of California - Irvine: Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php)

Specifically, [Online Retail Data set from United Kingdom](https://archive.ics.uci.edu/ml/datasets/online+retail)

## Start the Analysis


```python
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark.sql.window import Window
```


```python
spark = (SparkSession
         .builder
         .appName("RetentionExercise")
         .getOrCreate()
         )
```


```python
# Define schema
str_schema = StructType([
    StructField('InvoiceNo', IntegerType(), True),
    StructField('StockCode', StringType(), True),
    StructField('Description', StringType(), True),
    StructField('Quantity', FloatType(), True),
    StructField('InvoiceDate', StringType(), True),
    StructField('UnitPrice', FloatType(), True),
    StructField('CustomerID', IntegerType(), True),
    StructField('Country', StringType(), True)
])

# Pull data - download directly from online
#url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx"
# NOTE THIS IS NOT CSV FORMAT - NEED FORMAT CHANGES
#from pyspark import SparkFiles
#spark.sparkContext.addFile(url)
# customer_df = spark.read.csv("file://"+SparkFiles.get("online_retail.csv"), header = True, schema = str_schema)

# Pull data - if downloaded locally
customer_df = spark.read.csv("online_retail.csv", header = True, schema = str_schema)

# Change the date format to date format
customer_df = customer_df.withColumn("InvoiceDate", to_date(to_timestamp(col("InvoiceDate"), "M/d/yy H:mm")))
#customer_df = customer_df.withColumn("InvoiceDate", (to_timestamp(col("InvoiceDate"), "M/d/yy H:mm")))
```

# Data Evaluation


```python
# check the data
customer_df.show(3)
```

    +---------+---------+--------------------+--------+-----------+---------+----------+--------------+
    |InvoiceNo|StockCode|         Description|Quantity|InvoiceDate|UnitPrice|CustomerID|       Country|
    +---------+---------+--------------------+--------+-----------+---------+----------+--------------+
    |   536365|   85123A|WHITE HANGING HEA...|     6.0| 2010-12-01|     2.55|     17850|United Kingdom|
    |   536365|    71053| WHITE METAL LANTERN|     6.0| 2010-12-01|     3.39|     17850|United Kingdom|
    |   536365|   84406B|CREAM CUPID HEART...|     8.0| 2010-12-01|     2.75|     17850|United Kingdom|
    +---------+---------+--------------------+--------+-----------+---------+----------+--------------+
    only showing top 3 rows
    
    


```python
print(customer_df.printSchema())
```

    root
     |-- InvoiceNo: integer (nullable = true)
     |-- StockCode: string (nullable = true)
     |-- Description: string (nullable = true)
     |-- Quantity: float (nullable = true)
     |-- InvoiceDate: date (nullable = true)
     |-- UnitPrice: float (nullable = true)
     |-- CustomerID: integer (nullable = true)
     |-- Country: string (nullable = true)
    
    None
    

Examine date range


```python
customer_df.select(
    min("InvoiceDate").alias("Start Date"),
    max("InvoiceDate").alias("End Date")
).show()
```

    +----------+----------+
    |Start Date|  End Date|
    +----------+----------+
    |2010-12-01|2011-12-09|
    +----------+----------+
    
    

# Aggregation and Tabluation of data
Calculate relevant fields (date measure types) and aggregate the data


```python
test = (
    customer_df
    .select("*")
    .where(col("CustomerId").isNotNull() & col("InvoiceNo").isNotNull() & col("Country").isNotNull())
    .groupBy(["InvoiceDate", "CustomerId"])
    # By customers and by invoice date, aggregate up values
    .agg(round(sum(col("UnitPrice") * col("Quantity")),2).alias("total_value"))

    .orderBy(["CustomerId","InvoiceDate"])
)
test.show(6)
```

    +-----------+----------+-----------+
    |InvoiceDate|CustomerId|total_value|
    +-----------+----------+-----------+
    | 2011-01-18|     12346|   77183.59|
    | 2010-12-07|     12347|     711.79|
    | 2011-01-26|     12347|     475.39|
    | 2011-04-07|     12347|     636.25|
    | 2011-06-09|     12347|     382.52|
    | 2011-08-02|     12347|     584.91|
    +-----------+----------+-----------+
    only showing top 6 rows
    
    

I want to rank over in order of invoice date, for each customer ID


```python
window_count = Window.partitionBy(test['CustomerID']).orderBy(test['InvoiceDate'])

test = test.select('*', rank().over(window_count).alias('count_retention'))
test.show(6)
```

    +-----------+----------+-----------+---------------+
    |InvoiceDate|CustomerId|total_value|count_retention|
    +-----------+----------+-----------+---------------+
    | 2011-09-13|     12940|     361.27|              1|
    | 2011-10-16|     12940|     552.27|              2|
    | 2011-02-22|     13285|     666.89|              1|
    | 2011-04-27|     13285|     506.38|              2|
    | 2011-07-01|     13285|     796.83|              3|
    | 2011-11-16|     13285|     739.02|              4|
    +-----------+----------+-----------+---------------+
    only showing top 6 rows
    
    

Using the same schema, we create the last invoice date - the first order for new customers will be 'null'


```python
test = test.withColumn("InvoiceDate_last", lag("InvoiceDate", 1).over(window_count))
test.show(6)
```

    +-----------+----------+-----------+---------------+----------------+
    |InvoiceDate|CustomerId|total_value|count_retention|InvoiceDate_last|
    +-----------+----------+-----------+---------------+----------------+
    | 2011-09-13|     12940|     361.27|              1|            null|
    | 2011-10-16|     12940|     552.27|              2|      2011-09-13|
    | 2011-02-22|     13285|     666.89|              1|            null|
    | 2011-04-27|     13285|     506.38|              2|      2011-02-22|
    | 2011-07-01|     13285|     796.83|              3|      2011-04-27|
    | 2011-11-16|     13285|     739.02|              4|      2011-07-01|
    +-----------+----------+-----------+---------------+----------------+
    only showing top 6 rows
    
    

Calculate date difference - how many months since the last order
* Null (the first orders) will still be null - fill with zero


```python
test = test.withColumn("InvoiceDate_since_last", floor(months_between("InvoiceDate", "InvoiceDate_last"))).na.fill(0)
#.withColumn("InvoiceDate_since_last_date", floor(datediff("InvoiceDate", "InvoiceDate_last"))).na.fill(0) # If needing date difference, just in case
test.show(6)
```

    +-----------+----------+-----------+---------------+----------------+----------------------+
    |InvoiceDate|CustomerId|total_value|count_retention|InvoiceDate_last|InvoiceDate_since_last|
    +-----------+----------+-----------+---------------+----------------+----------------------+
    | 2011-09-13|     12940|     361.27|              1|            null|                     0|
    | 2011-10-16|     12940|     552.27|              2|      2011-09-13|                     1|
    | 2011-02-22|     13285|     666.89|              1|            null|                     0|
    | 2011-04-27|     13285|     506.38|              2|      2011-02-22|                     2|
    | 2011-07-01|     13285|     796.83|              3|      2011-04-27|                     2|
    | 2011-11-16|     13285|     739.02|              4|      2011-07-01|                     4|
    +-----------+----------+-----------+---------------+----------------+----------------------+
    only showing top 6 rows
    
    

Track date of first purchase for each customers


```python
test_first = (
    customer_df.groupBy(["CustomerId"]).agg(min(col("InvoiceDate")).alias("InvoiceDate_birth"))
)
test_first.show(2)
```

    +----------+-----------------+
    |CustomerId|InvoiceDate_birth|
    +----------+-----------------+
    |     17420|       2010-12-01|
    |     16861|       2010-12-06|
    +----------+-----------------+
    only showing top 2 rows
    
    


```python
df_fin = test.join(test_first, test["CustomerID"] == test_first["CustomerID"], "left")
df_fin = df_fin.withColumn("InvoiceDate_since_birth", floor(months_between("InvoiceDate", "InvoiceDate_birth")))
df_fin.show(10)
```

    +-----------+----------+-----------+---------------+----------------+----------------------+----------+-----------------+-----------------------+
    |InvoiceDate|CustomerId|total_value|count_retention|InvoiceDate_last|InvoiceDate_since_last|CustomerId|InvoiceDate_birth|InvoiceDate_since_birth|
    +-----------+----------+-----------+---------------+----------------+----------------------+----------+-----------------+-----------------------+
    | 2011-09-13|     12940|     361.27|              1|            null|                     0|     12940|       2011-09-13|                      0|
    | 2011-10-16|     12940|     552.27|              2|      2011-09-13|                     1|     12940|       2011-09-13|                      1|
    | 2011-02-22|     13285|     666.89|              1|            null|                     0|     13285|       2011-02-22|                      0|
    | 2011-04-27|     13285|     506.38|              2|      2011-02-22|                     2|     13285|       2011-02-22|                      2|
    | 2011-07-01|     13285|     796.83|              3|      2011-04-27|                     2|     13285|       2011-02-22|                      4|
    | 2011-11-16|     13285|     739.02|              4|      2011-07-01|                     4|     13285|       2011-02-22|                      8|
    | 2011-02-15|     13623|      156.0|              1|            null|                     0|     13623|       2011-02-15|                      0|
    | 2011-03-17|     13623|     101.35|              2|      2011-02-15|                     1|     13623|       2011-02-15|                      1|
    | 2011-04-03|     13623|     146.99|              3|      2011-03-17|                     0|     13623|       2011-02-15|                      1|
    | 2011-05-13|     13623|     147.58|              4|      2011-04-03|                     1|     13623|       2011-02-15|                      2|
    +-----------+----------+-----------+---------------+----------------+----------------------+----------+-----------------+-----------------------+
    only showing top 10 rows
    
    

# Tabular views - Retention
### Note that the counts for each month since are counted for unique customer IDs - for a typical retention curve, you would need to cumulate high to low
For example, if customers returned at 8th month since their first purchase, you can argue that the customers were 'active' at 7th months - we need to account for those columns for 7th month and less.

This may vary based on how you define 'active' customers

## (1) Retention of months since the first date of purchase


```python
tab_1 = (
    df_fin.groupBy("InvoiceDate_birth")
    .pivot("InvoiceDate_since_birth")
    .count()
    .orderBy(col("InvoiceDate_birth").asc())
).show(6)
```

    +-----------------+---+---+---+---+---+---+---+---+---+---+---+---+---+
    |InvoiceDate_birth|  0|  1|  2|  3|  4|  5|  6|  7|  8|  9| 10| 11| 12|
    +-----------------+---+---+---+---+---+---+---+---+---+---+---+---+---+
    |       2010-12-01|193| 90| 84| 86| 86| 96|102| 95| 90|114|103|147| 48|
    |       2010-12-02|154| 61| 57| 69| 55| 67| 65| 52| 61| 61| 59|101| 21|
    |       2010-12-03| 65| 30| 23| 32| 17| 33| 19| 29| 27| 18| 33| 40| 12|
    |       2010-12-05|114| 64| 39| 53| 44| 58| 49| 54| 45| 50| 58| 86| 13|
    |       2010-12-06|108| 34| 31| 43| 38| 48| 36| 44| 33| 40| 36| 66| 13|
    |       2010-12-07| 71| 26| 22| 32| 23| 31| 29| 34| 31| 34| 31| 54| 11|
    +-----------------+---+---+---+---+---+---+---+---+---+---+---+---+---+
    only showing top 6 rows
    
    

## (2) Retention - rolling purchase, accounting for recency of last purchase


```python
tab_2 = (
    df_fin.groupBy("InvoiceDate")
    .pivot("InvoiceDate_since_last")
    .count()
    .orderBy(col("InvoiceDate").desc())
)
tab_2.show(6)
```

    +-----------+---+---+---+----+----+----+----+----+----+----+----+----+----+
    |InvoiceDate|  0|  1|  2|   3|   4|   5|   6|   7|   8|   9|  10|  11|  12|
    +-----------+---+---+---+----+----+----+----+----+----+----+----+----+----+
    | 2011-12-09| 27|  5|  1|   1|null|null|   1|null|null|null|null|null|null|
    | 2011-12-08| 72| 11|  9|null|   3|null|   1|   5|   2|   2|null|null|null|
    | 2011-12-07| 66| 19|  5|   2|   1|null|null|   1|null|null|null|null|null|
    | 2011-12-06| 70| 16|  3|   2|   5|   2|null|null|   3|null|   1|   1|null|
    | 2011-12-05| 64| 18|  7|   5|   5|   1|   2|null|   1|   1|null|   1|null|
    | 2011-12-04| 34| 10|  6|   3|   1|null|null|   2|   1|null|   1|null|null|
    +-----------+---+---+---+----+----+----+----+----+----+----+----+----+----+
    only showing top 6 rows
    
    
