import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.sql.SQLContext
import org.apache.spark.sql.DataFrame

val sqlContext = new org.apache.spark.sql.SQLContext(sc)


val segments = sqlContext.read.format("com.databricks.spark.csv").option("delimiter", "\t").option("header","true").load("/home/neider/Documentos/PararellProgramming/train.tsv")

segments.show()
