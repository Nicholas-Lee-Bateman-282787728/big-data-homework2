import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.classification.{DecisionTreeClassifier, LinearSVC, LogisticRegression}
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
import org.apache.spark.ml.evaluation.{BinaryClassificationEvaluator, MulticlassClassificationEvaluator}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}

object Model {

  def main(args: Array[String]): Unit = {


    // before deploying to cluster we have trained the model
    val testOutputPath = "/home/abdurasul/output"   // path of testOutputFile
    val masterNodeAddress = "spark://10.91.54.103:7077"  // master node ip and port number
    val modelSavePath1 = "/home/abdurasul/decision-tree-cv" // path for the saved output of first model (in our case Decision Tree)
    val modelSavePath2 = "/home/abdurasul/svm-cv"    // path for the saved output of the second model (SVM)

    BasicConfigurator.configure()
    /*
      Creating Spark Session
      With a name Twitter sentiment Analysis
      With master node and get the session
     */
    val spark = SparkSession
      .builder()
      .appName("Twitter Sentiment Analysis")
      .master(masterNodeAddress)
      .getOrCreate()
    /*
      Creating the structure of the dataframe here id,text,label
     */
    val customSchema = StructType(Array(
      StructField("id", IntegerType, true),
      StructField("label", IntegerType, true),
      StructField("text", StringType, true))
    )

    /*

      Creating custom schema without lable colomn

     */
    val customTestSchema = StructType(Array(
      StructField("id", IntegerType, true),
      StructField("text", StringType, true)
    ))
    // here we will load the training data from dataset/train.csv
    // and we will not ignore the header
    val training = spark.read.format("csv").option("header", "true").schema(customSchema).load("dataset/train.csv")
    // as mentioned in report this is the part of the tokenizer that transforms text into words
    val tokenizer = new Tokenizer()
      .setInputCol("text")
      .setOutputCol("words")
    // feature extractor HashingTF
    // this will transform the sampels into features

    val hashingTF = new HashingTF()
      .setNumFeatures(1000)
      .setInputCol(tokenizer.getOutputCol)
      .setOutputCol("features")
    // we declare decisionTree classifier
    val dt = new DecisionTreeClassifier()
      .setMaxDepth(20)
    // declaration of the svm
    val lsvc = new LinearSVC()
      .setMaxIter(10)
      .setRegParam(0.1)
  // new pipeline having array of tokenizer, hashingTF and decision tree objects
    val pipeline1 = new Pipeline()
      .setStages(Array(tokenizer, hashingTF, dt))

    // We use a ParamGridBuilder to construct a grid of parameters to search over.
    // With 3 values for hashingTF.numFeatures and 2 values for lr.regParam,
    // this grid will have 3 x 2 = 6 parameter settings for CrossValidator to choose from.
    val paramGrid1 = new ParamGridBuilder()
      .addGrid(hashingTF.numFeatures, Array(10, 100, 1000, 2000))
      .addGrid(dt.maxDepth, Array(10, 20, 25))
      .build()

    // We now treat the Pipeline as an Estimator, wrapping it in a CrossValidator instance.
    // This will allow us to jointly choose parameters for all Pipeline stages.
    // A CrossValidator requires an Estimator, a set of Estimator ParamMaps, and an Evaluator.
    // Note that the evaluator here is a BinaryClassificationEvaluator and its default metric
    // is areaUnderROC.
    val cv = new CrossValidator()
      .setEstimator(pipeline1)
      .setEvaluator(new BinaryClassificationEvaluator)
      .setEstimatorParamMaps(paramGrid1)
      .setNumFolds(2)  // Use 3+ in practice
      .setParallelism(2)  // Evaluate up to 2 parameter settings in parallel

    // Run cross-validation, and choose the best set of parameters.
//    val cvModel = cv.fit(training)

    val pipeline2 = new Pipeline()
      .setStages(Array(tokenizer, hashingTF, lsvc))


    val paramGrid2 = new ParamGridBuilder()
      .addGrid(hashingTF.numFeatures, Array(1000, 2000))
      .addGrid(lsvc.maxIter, Array(25, 50))
      .build()


    // is areaUnderROC.
    val cv2 = new CrossValidator()
      .setEstimator(pipeline2)
      .setEvaluator(new BinaryClassificationEvaluator)
      .setEstimatorParamMaps(paramGrid2)
      .setNumFolds(2)  // Use 3+ in practice
      .setParallelism(2)  // Evaluate up to 2 parameter settings in parallel


//    val cvModel1 = cv.fit(training)
    val cvModel2 = cv2.fit(training)



//      cvModel1.write.overwrite().save(modelSavePath1)
      cvModel2.write.overwrite().save(modelSavePath2)
//    val test = spark.read.format("csv").option("header", "true").schema(customTestSchema).load("dataset/test.csv")

//    PipelineModel.load("dfd")

    import spark.implicits._
//    val tested1 = cvModel1.transform(training)
//      .select("prediction", "label")
//    val evaluator1 = new MulticlassClassificationEvaluator()
//      .setLabelCol("label")
//      .setPredictionCol("prediction")
//      .setMetricName("accuracy")
//    val accuracy1 = evaluator1.evaluate(tested1)
    // to write the outputfile after testing
    val tested2 = cvModel2.transform(training)
      .select("prediction", "label")
    val evaluator2 = new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")
    // having the accuracy value
    val accuracy2 = evaluator2.evaluate(tested2)

//    println("Accuracy Decision Tree: " + (accuracy1))
    println("Accuracy SVM: " + (accuracy2))


//    tested.repartition(10).write.csv(testOutputPath)

    spark.stop()
  }

  //  def stringify(c: Column) = concat(lit("["), concat_ws(",", c), lit("]"))

}
