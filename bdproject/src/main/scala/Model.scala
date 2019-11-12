import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.feature.{HashingTF, Tokenizer}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.{Column, Row}
import org.apache.spark.sql.types.{IntegerType, StringType, StructField, StructType}
import org.apache.spark.sql.SparkSession
import org.apache.log4j.BasicConfigurator
import org.apache.spark.sql.functions._
import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import org.apache.log4j.Logger

object Model {

  def main(args: Array[String]): Unit = {
    val testOutputPath = "/home/abdurasul/output"
    val masterNodeAddress = "spark://10.91.54.103:7077"
    val modelSavePath = "/home/abdurasul/decision-tree"

    BasicConfigurator.configure()
    val spark = SparkSession
      .builder()
      .appName("Twitter Sentiment Analysis")
      .master(masterNodeAddress)
      .getOrCreate()

    val customSchema = StructType(Array(
      StructField("id", IntegerType, true),
      StructField("label", IntegerType, true),
      StructField("text", StringType, true))
    )

    val customTestSchema = StructType(Array(
      StructField("id", IntegerType, true),
      StructField("text", StringType, true)
    ))

    val training = spark.read.format("csv").option("header", "true").schema(customSchema).load("dataset/train.csv")

    val tokenizer = new Tokenizer()
      .setInputCol("text")
      .setOutputCol("words")
    val hashingTF = new HashingTF()
      .setNumFeatures(1000)
      .setInputCol(tokenizer.getOutputCol)
      .setOutputCol("features")
    val lr = new LogisticRegression()
      .setMaxIter(10)
      .setRegParam(0.001)
    val dt = new DecisionTreeClassifier()
      .setMaxDepth(20)

    val pipeline = new Pipeline()
      .setStages(Array(tokenizer, hashingTF, dt))

    val model = pipeline.fit(training)


    model.write.overwrite().save(modelSavePath)

    val test = spark.read.format("csv").option("header", "true").schema(customTestSchema).load("dataset/test.csv")

    import spark.implicits._
    val tested = model.transform(test)
      .select("id", "text", "prediction")


    tested.repartition(10).write.csv(testOutputPath)

    spark.stop()
  }

  //  def stringify(c: Column) = concat(lit("["), concat_ws(",", c), lit("]"))

}
