# project-6
working with and querying steam data and games data to answer questions based on the datasets  on databricks community edition using the machine learning work page to create predictions and recommendation models
# Databricks notebook source
# data preparation

import mlflow

mlflow.pyspark.ml.autolog()

# COMMAND ----------

steam = spark.read.csv("/FileStore/tables/steam_200k.csv",
                   header = "true",
                   inferSchema="true")

# COMMAND ----------

steam = steam.dropna()
steam.show()

# COMMAND ----------

steam.printSchema()

# COMMAND ----------

from pyspark.sql.functions import col

steam = steam.withColumn("hoursp/bought", col("hoursp/bought").cast("integer"))

# COMMAND ----------

steam.printSchema()

# COMMAND ----------

games = spark.read.csv("/FileStore/tables/games.csv",
                   header = "true",
                   inferSchema="true")

# COMMAND ----------

games = games.dropna()
games.show()

# COMMAND ----------

games.printSchema()

# COMMAND ----------

steam.show(truncate=False)

# COMMAND ----------

games.show(truncate=False)

# COMMAND ----------


from pyspark.sql.functions import monotonically_increasing_id

games_df = spark.read.csv("/FileStore/tables/games.csv", header=True)
steam = spark.read.csv("/FileStore/tables/steam_200k.csv", header=True)

games_with_unique_id = games.withColumn("unique_game_id", monotonically_increasing_id())

# Joining datasets
merged_df = steam.join(games_with_unique_id, steam["gameplayed"] == games_with_unique_id["games"], "inner")


merged_df.show()

# COMMAND ----------

merged_df.printSchema()

# COMMAND ----------

from pyspark.sql.functions import col


merged_df = merged_df.withColumn("user-id", merged_df["user-id"].cast("integer"))

# COMMAND ----------

from pyspark.sql.functions import col


merged_df = merged_df.withColumn("hoursp/bought", merged_df["hoursp/bought"].cast("integer"))

# COMMAND ----------

merged_df.printSchema()

# COMMAND ----------

# Splitting the data into training and test datasets
train_data, test_data = merged_df.randomSplit([0.8, 0.2], seed=100)


merged_train_data = train_data.unionAll(merged_df)

# COMMAND ----------

# MAGIC %md
# MAGIC training the model 

# COMMAND ----------

from pyspark.ml.recommendation import ALS

als = ALS(
    userCol="user-id",
    itemCol="unique_game_id",
    ratingCol="hoursp/bought",
    coldStartStrategy="drop"
)

model = als.fit(merged_df)

# COMMAND ----------

predictions = model.transform(test_data).dropna()
predictions.show()

# COMMAND ----------

from pyspark.ml.evaluation import RegressionEvaluator


evaluator = RegressionEvaluator(labelCol="hoursp/bought", predictionCol="prediction", metricName="rmse")


rmse = evaluator.evaluate(predictions)


print('Root Mean Squared Error is %g' %rmse)

# COMMAND ----------

# MAGIC %md
# MAGIC generating recommendations

# COMMAND ----------

from pyspark.sql import Row
from pyspark.sql import functions

predictions_df = merged_df.withColumn("userId", functions.expr("int('0')"))


predictions_df = model.transform(predictions_df)


sorted_recommendations = predictions_df.orderBy(predictions_df["prediction"].desc())


sorted_recommendations.show()
