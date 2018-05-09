#Neider Alejandro Fajardo-20142020025
#Ejercicio Machine Learning que incluye pipeline y corssValidation
#utilizando un método de dessicionTree

from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.feature import StringIndexer, VectorIndexer,HashingTF
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
from pyspark.context import SparkContext
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql.session import SparkSession

sc = SparkContext('local')
spark = SparkSession(sc)

# Load the data stored in LIBSVM format as a DataFrame.
data = spark.read.format("libsvm").load("/home/neider/Documentos/PararellProgramming/MachineLearning1/sample_binary_classification_data.txt")

# Index labels, adding metadata to the label column.
# Fit on whole dataset to include all labels in index.
labelIndexer = StringIndexer(inputCol="label", outputCol="indexedLabel").fit(data)
# Automatically identify categorical features, and index them.
# We specify maxCategories so features with > 4 distinct values are treated as continuous.
featureIndexer =\
    VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=4).fit(data)

# Split the data into training and test sets (30% held out for testing)
(trainingData, testData) = data.randomSplit([0.7, 0.3])

# Train a DecisionTree model.
dt = DecisionTreeClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures")

# Chain indexers and tree in a Pipeline
pipeline = Pipeline(stages=[labelIndexer, featureIndexer, dt])

# Train model.  This also runs the indexers.
model = pipeline.fit(trainingData)

# Make predictions.
predictions = model.transform(testData)

# Select (prediction, true label) and compute test error
evaluator = MulticlassClassificationEvaluator(
    labelCol="indexedLabel", predictionCol="prediction", metricName="accuracy")


treeModel = model.stages[2]
# summary only
print(treeModel)
hashingTF = HashingTF(inputCol="features", outputCol="features")
regParam = 0.3
paramGrid = ParamGridBuilder() \
    .addGrid(hashingTF.numFeatures, [10, 100, 1000]) \
    .addGrid(regParam, [0.1, 0.01]) \
    .build()

crossval = CrossValidator(estimator=pipeline,
                          estimatorParamMaps=paramGrid,
                          evaluator=evaluator,
                          numFolds=2)

cvModel = crossval.fit(trainingData)

prediction = cvModel.transform(testData)
prediction.select("indexedLabel","prediction").show()

accuracy = evaluator.evaluate(predictions)
print("Test Error = %g " % (1.0 - accuracy))
