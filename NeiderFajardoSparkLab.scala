//Neider Alejandro Fajardo 20142020025
//Import de sql Context
val sqlContext = new org.apache.spark.sql.SQLContext(sc)
//Load data, Inside the quotation marks place the location of the file type Parquet
val df = sqlContext.read.parquet("/home/neider/Documentos/PararellProgramming/SparkExercises/ejercicio1.parquet")


//--------------------RDD transformations and actions:
//First the data type is converted from sql to rdd
val datosRDD = sc.parallelize(df.collect())

//Filter out only the flights arriving from USA.
val USAarriving = datosRDD.filter(_(1).toString().equalsIgnoreCase("United States"))
USAarriving.collect()

//Find the min count of flights departing for USA.
val minToUSA = datosRDD.filter(_(0).toString.equalsIgnoreCase("United States")).map(x => x.getLong(2)).reduce((a, b) => if (a < b) a else b)

//Find the max count of flights departing for any country.
val maxCount = datosRDD.map(x => x.getLong(2)).reduce((a, b) => if (a > b) a else b)

//Find the total of all flights by origin country.
val totalOriginCountry = datosRDD.map(x => (x(1).toString, x.getLong(2))).reduceByKey((a, b) => a + b)
totalOriginCountry.collect()

//Find the total of all flights by destination country.
val totalDestination = datosRDD.map(x => (x(0).toString,x.getLong(2))).reduceByKey((a, b) => a + b)
totalDestination.collect()

//Find the total of all flights departing from countries that begin with letter 'S'
val totalFlightsDepartingS = datosRDD.filter(_(1).toString().startsWith("S")).map(x => (x.getLong(2))).reduce((a, b) => a + b)


//--------------------Dataframes SQL:
//First convert data to temporal table call data
df.registerTempTable("data")

//Filter out only the flights arriving from USA.
val USAarrivingSQL = sqlContext.sql("SELECT * FROM data WHERE ORIGIN_COUNTRY_NAME = 'United States'")
USAarrivingSQL.show()

//Find the min count of flights departing for USA.
val minVal = sqlContext.sql("SELECT min(count) FROM data WHERE DEST_COUNTRY_NAME = 'United States'")
minVal.show()

//Find the max count of flights departing for any country.
val maxVal = sqlContext.sql("SELECT max(count) FROM data")
maxVal.show()

//Find the total of all flights by origin country.
val totalOriginCountry = sqlContext.sql("SELECT sum(count), ORIGIN_COUNTRY_NAME  FROM data group by ORIGIN_COUNTRY_NAME")
totalOriginCountry.show()

//Find the total of all flights by destination country.
val totalDestinationCountry = sqlContext.sql("SELECT sum(count), DEST_COUNTRY_NAME  FROM data group by DEST_COUNTRY_NAME")
totalDestinationCountry.show()

//Find the total of all flights departing from countries that begin with letter 'S'
val totalFlightsDepartingS = sqlContext.sql("SELECT sum(count) FROM data WHERE lower(ORIGIN_COUNTRY_NAME) like 's%'")
totalFlightsDepartingS.show()


//--------------------Dataframes API:
//Covert data to RDD
val dataRDD = sc.parallelize(df.collect())

//Convert data from RDD to Dataframe
val dataFrame = df.sqlContext.createDataFrame(dataRDD, df.schema)

//Filter out only the flights arriving from USA.
dataFrame.filter(dataFrame("ORIGIN_COUNTRY_NAME")==="United States").show()

//Find the min count of flights departing for USA.
val minFromUSA =dataFrame.filter(dataFrame("ORIGIN_COUNTRY_NAME")==="United States").map(x => x.getLong(2)).reduce{(a , b) => if (a < b) a else b}

//Find the max count of flights departing for any country.
val maxFromAny = dataFrame.map(x => x.getLong(2)).reduce{(a,b) => if (a > b) a else b}

//Find the total of all flights by origin country.
val aux = dataFrame.groupBy("ORIGIN_COUNTRY_NAME").sum("count").show

//Find the total of all flights by destination country.
val departing = dataFrame.groupBy("DEST_COUNTRY_NAME").sum("count").show

//Find the total of all flights departing from countries that begin with letter 'S'
dataFrame.filter(dataFrame("ORIGIN_COUNTRY_NAME").startsWith("S")).map(x => x.getLong(2)).reduce{(a,b) => a+b}
