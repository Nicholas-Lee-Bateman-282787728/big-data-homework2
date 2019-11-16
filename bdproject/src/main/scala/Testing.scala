package org.apache.spark.examples.ml
import org.apache.spark.ml.Transformer
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.types._
import java.util.Locale

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
    val masterNodeAddress = "spark://10.91.54.103:7077"
    val modeldirectory = "/home/abdurasul/decision-tree"
    val testOutputPath = "/home/abdurasul/output-spark4"
    val spark = SparkSession
      .builder()
      .appName("Twitter Sentiment Analysis Testing")
      .master(masterNodeAddress)
      .getOrCreate()

    val customSchema = StructType(Array(
      StructField("id", IntegerType, true),
      StructField("label", DoubleType, true),
      StructField("text", StringType, true))
    )


    import spark.implicits._
    val model = org.apache.spark.ml.PipelineModel.load(modeldirectory)
    val training = spark.read.format("csv").option("header", "true").schema(customSchema).load("dataset/train.csv")

//    evaluateClassificationModel(model, training, "label")

//    val out = model.transform(training)
//      .select("id", "text", "prediction", "label")
    implicit def bool2int(b:Boolean) = if (b) 1 else 0
    val pl = model.transform(training)
    .select("prediction", "label")
    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")
    val accuracy = evaluator.evaluate(pl)
    println("Test Error = " + (1.0 - accuracy))



    spark.stop()
//    val ssc = new StreamingContext(spark.sparkContext, Seconds(1))
//    val lines = ssc.socketTextStream("10.91.66.168", 8998)
//    lines.print(1)
//    val df1 = spark.readStream.format("kafka").option("kafka.bootstrap.servers", "10.91.66.168:8989").load()

//    println("-----", lines.toString())
//    ssc.start()
//    ssc.awaitTermination()
//    println(df1.printSchema)



  }
//  def evaluateClassificationModel(
//                                               model: Transformer,
//                                               data: DataFrame,
//                                               labelColName: String): Unit = {
//    import org.apache.spark.ml.util.MetadataUtils
//    val fullPredictions = model.transform(data).cache()
//    val predictions = fullPredictions.select("prediction").rdd.map(_.getDouble(0))
//    println("predictions ------", predictions)
//    val labels = fullPredictions.select(labelColName).rdd.map(_.getDouble(0))
//    println("labels ------", labels)
//    // Print number of classes for reference.
//
////    val numClasses = MetadataUtils.getNumClasses(fullPredictions.schema(labelColName)) match {
////      case Some(n) => n
////      case None => throw new RuntimeException(
////        "Unknown failure when indexing labels for classification.")
////    }
//
//    val accuracy = new MulticlassMetrics(predictions.zip(labels)).accuracy
//    println(s"  Accuracy (2 classes): $accuracy")
//  }
}
