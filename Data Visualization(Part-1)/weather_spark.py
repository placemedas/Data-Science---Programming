# This program creates three output files. These files are required for weather_plot.py to run.
# 1. stat.csv
# 2. global_pred.csv
# 3. test_pred.csv
# Input files used for Part B - Task a and b1 are tmax-2 and for b2 is tmax-test
import sys
import elevation_grid as eg
import pandas as pd
import numpy as np
import datetime
assert sys.version_info >= (3, 5)  # make sure we have Python 3.5+
from numpy import array
from pyspark.sql import SparkSession, functions, types
from pyspark.sql.functions import lit,floor
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, SQLTransformer
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.regression import GBTRegressor
from pyspark.ml import PipelineModel

spark = SparkSession.builder.appName('weather prediction').getOrCreate()
spark.sparkContext.setLogLevel('WARN')
assert spark.version >= '2.4'  # make sure we have Spark 2.4+

# Defining global variable here.

# Defining the schema for tmax datasets
def weather_schema():
    tmaxschema = types.StructType([
        types.StructField('station', types.StringType()),
        types.StructField('date', types.DateType()),
        types.StructField('latitude', types.FloatType()),
        types.StructField('longitude', types.FloatType()),
        types.StructField('elevation', types.FloatType()),
        types.StructField('tmax', types.FloatType()),
    ])
    return tmaxschema


# Calculate the difference in avg temperature between two 50 year time periods ie. 1977-'96 and 1997 -2017
def temp_max(input):
    tmax_schema = weather_schema()
    # Spark read of data
    temp = spark.read.csv(input, schema=tmax_schema).cache()
    # Remove rows that do not have temperature
    temp = temp.filter(temp["tmax"].isNotNull())

    # Prepare two dataframes that has average temperatures for 50 years
    temp.createOrReplaceTempView("temp")
    stat20 = spark.sql(
        "select station,latitude,longitude,avg(tmax) as tmax1 from temp where year(date) >= 1997 and year(date) <= 2017 group by station,latitude,longitude").cache()
    stat19 = spark.sql(
        "select station,latitude,longitude,avg(tmax) as tmax2 from temp where year(date) >= 1977 and year(date) <= 1996  group by station,latitude,longitude").cache()

    stat20.createOrReplaceTempView("stat20")
    stat19.createOrReplaceTempView("stat19")

    # Calculate the difference to understand the change in climate
    stats = spark.sql(
        "select a.latitude,a.longitude,(a.tmax1 - b.tmax2) as tdiff from stat20 a join stat19 b on a.station = b.station "
        "and a.latitude = b.latitude and a.longitude = b.longitude").cache()

    # Categorize the temperature difference into multiple classes
    stats.createOrReplaceTempView("stats")
    stats_f = spark.sql("select *,CASE WHEN tdiff >= 9 THEN '+9° & above' "
                        "WHEN tdiff >= 7 AND tdiff < 9 THEN '+7° - +8°' "
                        "WHEN tdiff >= 4 AND tdiff < 7 THEN '+4° - +6°' "
                        "WHEN tdiff >= 2 AND tdiff < 4 THEN '+2° - +3°' "
                        "WHEN tdiff >= 1 AND tdiff < 2 THEN '+1°' "
                        "WHEN tdiff >= 0 AND tdiff < 1 THEN '+0°' "
                        "WHEN tdiff >= -1 AND tdiff < 0 THEN '-1°' "
                        "WHEN tdiff >= -3 AND tdiff < -1 THEN '-2° to -3°' "
                        "WHEN tdiff >= -6 AND tdiff < -3 THEN '-4° to -6°' "
                        "WHEN tdiff >= -8 AND tdiff < -6 THEN '-7° to -8°' "
                        # "WHEN tdiff >= -9 AND tdiff < -7 THEN '-8° to -9°' "
                        "else '<= -9°' end as category "
                        "from stats ")
    # stats_f.show()
    return stats_f


# Model creation
def model_creator(input, model_file):
    tmax_schema = weather_schema()
    temp = spark.read.csv(input, schema=tmax_schema)
    # Training and test split
    train, validation = temp.randomSplit([0.75, 0.25], seed=70)
    train = train.cache()
    validation = validation.cache()
    train.registerTempTable("train")
    # Query with out using yesterday max
    query = "SELECT latitude,longitude,elevation,DAYOFYEAR(date) as day,tmax as tmax FROM __THIS__"

    temp_sqltrans = SQLTransformer(statement=query)

    # Vector Assembler with out using yesterday max
    temp_assembler = VectorAssembler(inputCols=["latitude", "longitude", "elevation", "day"], outputCol="features")

    # Fitting Models
    regressor = GBTRegressor(featuresCol="features", labelCol='tmax', predictionCol='prediction')

    # Model evaluation
    evaluator = RegressionEvaluator(labelCol="tmax", predictionCol="prediction", metricName="rmse")

    temp_pipeline = Pipeline(stages=[temp_sqltrans, temp_assembler, regressor])
    tmax_model = temp_pipeline.fit(train)
    tmax_prediction = tmax_model.transform(validation)

    # RMSE and R2 score calculation
    rmse_score = evaluator.evaluate(tmax_prediction)

    evaluator_r2 = RegressionEvaluator(labelCol="tmax", predictionCol="prediction", metricName="r2")
    r2_score = evaluator_r2.evaluate(tmax_prediction)

    print('Validation  RMSE score for TMAX model: %g' % (rmse_score,))
    print('Validation  R2 score for TMAX model: %g' % (r2_score,))

    # Saving the model
    tmax_model.write().overwrite().save(model_file)

#Function to perform prediction for global latitude and longitude
def global_pred(model_file):
    #Create a numpy array with all latitude and longitude and then obtain its elevation
    lats, lons = np.meshgrid(np.arange(-90,90,.5),np.arange(-180,180,.5))
    elevs = [eg.get_elevations(np.array([late,lone]).T) for late,lone in zip(lats,lons)]

    #Conversion of latitude and longitude into vector
    lat = [float(lt) for lt in array(lats).flat]
    lon = [float(ln) for ln in array(lons).flat]
    elev = [float(el) for el in array(elevs).flat]

    #Dataframe creation with all columns to be passed for prediction. Prediction is done for 2020-01-20
    df_global = spark.createDataFrame(zip(lat, lon, elev), schema=['latitude', 'longitude', 'elevation'])
    df_global = df_global.withColumn("station",lit('GLOB100'))
    date1 = datetime.datetime.strptime('2020-01-20', '%Y-%m-%d').date()
    df_global = df_global.withColumn("date",lit(date1))
    df_global = df_global.withColumn("tmax",lit(float(0.0)))
    df_global = df_global.select("station","date","latitude","longitude","elevation","tmax")

    #Prediction using saved model
    model = PipelineModel.load(model_file)
    predictions = model.transform(df_global)
    predictions = predictions.select("latitude","longitude","elevation","tmax","prediction")
    #predictions.show()
    return predictions


def test_model(model_file, inputs):
    tmax_schema = weather_schema()
    # get the data
    test_tmax = spark.read.csv(inputs, schema=tmax_schema)
    test_tmax.createOrReplaceTempView("test_tmax")
    test_tmax = spark.sql("select * from test_tmax where year(date) = 2015")
    test_tmax = test_tmax.filter(test_tmax["tmax"].isNotNull())

    # load the model
    model = PipelineModel.load(model_file)

    # use the model to make predictions
    test_output = model.transform(test_tmax)

    # Inclusion of regression error
    test_output = test_output.withColumn("R_Error", floor(test_output["prediction"] - test_output["tmax"]))

    # Below code groups those latitude and longitude and obtains average error. This avoids overplotting
    test_output.createOrReplaceTempView("test_out")
    test_output = spark.sql(
        "select latitude,longitude,avg(R_Error) as R_Error from test_out group by latitude,longitude")

    # Categorizing regression for the easiness to plot
    test_output.createOrReplaceTempView("test")
    test_output = spark.sql("select *,CASE WHEN R_Error >= 9 THEN '+9° & above' "
                            "WHEN R_Error >= 7 AND R_Error < 9 THEN '+7° - +8°' "
                            "WHEN R_Error >= 4 AND R_Error < 7 THEN '+4° - +6°' "
                            "WHEN R_Error >= 2 AND R_Error < 4 THEN '+2° - +3°' "
                            "WHEN R_Error >= 1 AND R_Error < 2 THEN '+1°' "
                            "WHEN R_Error >= 0 AND R_Error < 1 THEN '+0°' "
                            "WHEN R_Error >= -1 AND R_Error < 0 THEN '-1°' "
                            "WHEN R_Error >= -3 AND R_Error < -1 THEN '-2° to -3°' "
                            "WHEN R_Error >= -6 AND R_Error < -3 THEN '-4° to -6°' "
                            "WHEN R_Error >= -8 AND R_Error < -6 THEN '-7° to -8°' "
                            # "WHEN tdiff >= -9 AND tdiff < -7 THEN '-8° to -9°' "
                            "else '<= -9°' end as category "
                            "from test ")
    return test_output

def main():
    #### Part 2 - Task a preprocessing
    # Function to prepare the temperature comparison dataset
    input = "tmax-2"
    stats_result = temp_max(input)
    # Conversion of spark df to pandas
    stat = stats_result.toPandas()
    stat.to_csv('stat.csv')

    #### Part 2 - Task b1 preprocessing
    dataset = "tmax-2"
    model_file = "weather_model"
    # Calling the function to create the model
    model_creator(dataset, model_file)

    ####Part 2 - Task b2 -1 prediction on global co-ordinates
    predictions = global_pred(model_file)
    # Conversion of predicted results to pandas
    result_df1 = predictions.toPandas()
    result_df1.to_csv('global_pred.csv')

    ####Part 2 - Task b2-2 prediction on test dataset
    test_data = "tmax-test"
    # Prediction function test_model
    test_predictions = test_model(model_file, test_data)
    # Conversion of spark df to pandas
    t_pred = test_predictions.toPandas()
    t_pred.to_csv('test_pred.csv')

if __name__ == '__main__':
    main()