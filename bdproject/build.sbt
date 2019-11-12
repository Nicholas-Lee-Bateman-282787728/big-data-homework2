name := "bdproject"

version := "0.1"

scalaVersion := "2.11.12"

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % "2.4.4",
  "org.apache.spark" %% "spark-sql" % "2.4.4",
  "org.apache.spark" %% "spark-mllib" % "2.4.4",
  "log4j" % "log4j" % "1.2.17",
  "com.databricks" %% "spark-avro" % "3.2.0",
  "com.google.guava" % "guava" % "15.0"
)