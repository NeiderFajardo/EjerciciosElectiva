
from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
from pyspark.sql import SQLContext

sc = SparkContext('local')
spark = SparkSession(sc)
sql = SQLContext(sc)
train = spark.read.format("com.databricks.spark.csv").option("delimiter", "\t").load("/home/neider/Documentos/PararellProgramming/train.tsv")
test = spark.read.format("com.databricks.spark.csv").option("delimiter", "\t").load("/home/neider/Documentos/PararellProgramming/test.tsv")
df_train = train.selectExpr("_c0 as train_id","_c1 as name","_c2 as item_condition_id","_c3 as category_name","_c4 as brand_name","_c5 as price","_c6 as shipping","_c7 as item_description")
df_test = test.selectExpr("_c0 as test_id","_c1 as name","_c2 as item_condition_id","_c3 as category_name","_c4 as brand_name","_c5 as shipping","_c6 as item_description")


df_train =df_train[df_train.name != "name"]
df_test =df_test[df_test.name != "name"]
#valores =df_train[df_train.brand_name == "Electronics/Video Games & Consoles/Games"]
#valores =df_train.filter(df_train.category_name.like('%Electronics%') | df_train.category_name.like('%Video Games & Consoles%') | df_train.category_name.like('%Games%'))
valores = df_train[df_train.category_name == "Electronics/Video Games & Consoles/Games"]

valores.show(20)
print(valores.count())
df_test.show(20)

print("Dimension of Train:",str(df_train.count()))
print("Name of the files Train: ",df_train.columns)

print("Dimension of data Test:",str(df_test.count()))
print("Name of the files Test: ",df_test.columns)

noValue= df_train[df_train.category_name != "null"]
print("The number of items without brand: ",str(noValue.count()))
print("The number of items with brand: ",str(df_train.count()-noValue.count()))
