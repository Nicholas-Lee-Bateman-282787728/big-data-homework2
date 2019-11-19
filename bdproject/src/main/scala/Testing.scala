package org.apache.spark.examples.ml
import org.apache.spark.ml.Transformer
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.sql.{DataFrame, SaveMode, SparkSession}
import org.apache.spark.sql.types._
import java.util.{Calendar, Locale}

import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator

import scala.collection.mutable

//import scopt.OptionParser

//import org.apache.spark.examples.mllib.AbstractParams
import org.apache.spark.ml.{Pipeline, PipelineStage, Transformer}
import org.apache.spark.ml.classification.{DecisionTreeClassificationModel, DecisionTreeClassifier}
import org.apache.spark.ml.feature.{StringIndexer, VectorIndexer}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.regression.{DecisionTreeRegressionModel, DecisionTreeRegressor}
import org.apache.spark.ml.util.MetadataUtils
import org.apache.spark.mllib.evaluation.{MulticlassMetrics, RegressionMetrics}
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.sql.{DataFrame, SparkSession}

import org.apache.spark.sql.functions._
import org.apache.spark.streaming.{Seconds, StreamingContext}

object Testing {
  def main(args: Array[String]): Unit = {
    
    val masterNodeAddress = "spark://10.91.54.103:7077" // master node ip:port address
    val modeldirectory = "decision-tree" // the output of the first classifier which is in our case the decision tree
    val testOutputPath = "output-spark"// the output of the second classifier which is in our case the svm
    val streamOutputPath = "stream_output"

    // creating spark session
    val spark = SparkSession
      .builder()   // build the spark session
      .appName("Twitter Sentiment Analysis Testing") // name the spark session 
      .master(masterNodeAddress) // given the master ip:port adress
      .getOrCreate() // to create

    val customSchema = StructType(Array(  
      StructField("id", IntegerType, true),   // the column name id
      StructField("label", DoubleType, true), // label 0,1 if negative, positive
      StructField("text", StringType, true)) // and the comment itself
    )


    import spark.implicits._
    val model = org.apache.spark.ml.PipelineModel.load(modeldirectory)   /// here we have new Pipeline to load the model
    val testing = spark.read.format("csv").option("header", "true").schema(customSchema).load("dataset/testing.csv")   // and here we are loading the testing.csv data

//    evaluateClassificationModel(model, testing, "label")

//    val out = model.transform(testing)
//      .select("id", "text", "prediction", "label")
    implicit def bool2int(b:Boolean) = if (b) 1 else 0    /// a method that integarize a binarized value from true->1 and from false->0
    val pl = model.transform(testing)    /// now  we are calculating the labels for the dataset testing.csv
    .select("prediction", "label")
    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")
    val accuracy = evaluator.evaluate(pl)
    println("Test Error = " + (1.0 - accuracy))

    var id = 1

    val ssc = new StreamingContext(spark.sparkContext, Seconds(30));  // creating spark streaming context
    val lines = ssc.socketTextStream("10.90.138.32", 8989)  // reading from the stream of the host
    lines.print(1)
    lines.foreachRDD(foreachFunc = (rdd, time) =>{      //for each tweeet, convert inputsocstream to RDD,
      if (rdd.collect().length != 0) {
        val df = Seq((id, rdd.toString())).toDF("id", "text")      // creating Dataframe
        val predicted = model.transform(df)                      // prediction of the text
        val label = predicted.select("prediction").first().getDouble(0)   // Get the labels
        val time_now = Calendar.getInstance().getTime()                 // get the time
        val outDF = Seq((id, time_now.toString(), rdd.first(), label)).toDF("id", "time", "text", "label") //to write the file
        outDF.write.mode(SaveMode.Append).csv(streamOutputPath)  // in the output path
        println(predicted.select("text", "prediction"))
        id += 1
      }
    })


  }

}
