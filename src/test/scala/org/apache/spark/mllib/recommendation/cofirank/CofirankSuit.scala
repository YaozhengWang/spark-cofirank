package org.apache.spark.mllib.recommendation.cofirank

import org.apache.spark.{SparkContext, SparkConf}

/**
 * Created by cage on 2016/6/22.
 */
object CofirankSuit {
  def main(args: Array[String]) = {
    val conf = new SparkConf().setAppName("MovieLensCofirank").setMaster("spark://node-6:7077").setJars(List("D:\\IdeaProjects\\spark-cofirank\\out\\artifacts\\spark_cofirank_jar\\spark-cofirank.jar"))
    val sc = new SparkContext(conf)
    val ratings = sc.textFile("hdfs://node-6:9000/testdata/ratings.dat").map{ line =>
      val fields = line.split("::")
      Rating(fields(0).toInt, fields(1).toInt, fields(2).toDouble)
    }
    val rank = 10
    val numIter = 10
    val lambda = 1.0
    val model = Cofirank.train(ratings, rank, numIter, lambda)
   // model.save(sc, "hdfs://node-6:9000/testdata")
  }
}
