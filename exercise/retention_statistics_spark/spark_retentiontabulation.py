# -*- coding: utf-8 -*-
"""spark_RetentionTabulation.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1vEx0MWOqA9y__HbYr-DdVMoKbj7hEMGm

# Spark exercise - Handling a large retail data
This workflow assumes that you have installed Apache Spark and pyspark successfully, along with internet access to acquire data used.

The data is pulled from [University of California - Irvine: Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php)

Specifically, [Online Retail Data set from United Kingdom](https://archive.ics.uci.edu/ml/datasets/online+retail)

## Start the Analysis
"""

from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark.sql.window import Window

spark = (SparkSession
         .builder
         .appName("RetentionExercise")
         .getOrCreate()
         )

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

"""# Data Evaluation"""

# check the data
customer_df.show(3)

print(customer_df.printSchema())

"""Examine date range"""

customer_df.select(
    min("InvoiceDate").alias("Start Date"),
    max("InvoiceDate").alias("End Date")
).show()

"""# Aggregation and Tabluation of data
Calculate relevant fields (date measure types) and aggregate the data
"""

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

"""I want to rank over in order of invoice date, for each customer ID"""

window_count = Window.partitionBy(test['CustomerID']).orderBy(test['InvoiceDate'])

test = test.select('*', rank().over(window_count).alias('count_retention'))
test.show(6)

"""Using the same schema, we create the last invoice date - the first order for new customers will be 'null'"""

test = test.withColumn("InvoiceDate_last", lag("InvoiceDate", 1).over(window_count))
test.show(6)

"""Calculate date difference - how many months since the last order
* Null (the first orders) will still be null - fill with zero
"""

test = test.withColumn("InvoiceDate_since_last", floor(months_between("InvoiceDate", "InvoiceDate_last"))).na.fill(0)
#.withColumn("InvoiceDate_since_last_date", floor(datediff("InvoiceDate", "InvoiceDate_last"))).na.fill(0) # If needing date difference, just in case
test.show(6)

"""Track date of first purchase for each customers"""

test_first = (
    customer_df.groupBy(["CustomerId"]).agg(min(col("InvoiceDate")).alias("InvoiceDate_birth"))
)
test_first.show(2)

df_fin = test.join(test_first, test["CustomerID"] == test_first["CustomerID"], "left")
df_fin = df_fin.withColumn("InvoiceDate_since_birth", floor(months_between("InvoiceDate", "InvoiceDate_birth")))
df_fin.show(10)

"""# Tabular views - Retention
### Note that the counts for each month since are counted for unique customer IDs - for a typical retention curve, you would need to cumulate high to low
For example, if customers returned at 8th month since their first purchase, you can argue that the customers were 'active' at 7th months - we need to account for those columns for 7th month and less.

This may vary based on how you define 'active' customers

## (1) Retention of months since the first date of purchase
"""

tab_1 = (
    df_fin.groupBy("InvoiceDate_birth")
    .pivot("InvoiceDate_since_birth")
    .count()
    .orderBy(col("InvoiceDate_birth").asc())
).show(6)

"""## (2) Retention - rolling purchase, accounting for recency of last purchase"""

tab_2 = (
    df_fin.groupBy("InvoiceDate")
    .pivot("InvoiceDate_since_last")
    .count()
    .orderBy(col("InvoiceDate").desc())
)
tab_2.show(6)