#Cofirank for Spark: a Recommendation based on Ranking for Apache Spark
This package is an implementation of the Cofirank algorithm for Apache Spark.

The original Cofirank algorithm is provided by Markus Weimer[1], which based on the evaluation index of NDCG@k.
The paper considers that it is much more important to rank the first items right than the last ones. In other
words, it is more important to predict what a user likes than what she dislikes. All of above reasons lead to 
the following goals:
* The algorithm needs to be able to optimize ranking scores directly[2].
* The algorithm should not require any features besides the actual ratings.
* The algorithm needs to scale well and parallelize such as to deal with millions of ratings arising from thousands of items and users.

In this project, the Cofirank is parallelized by Spark, which is edited by **IntelliJ IDEA Community Edition 14.1.2**.
To report issues or request features about TFOCS for Spark, please use our GitHub issues page.

##Usage Example
```
package org.apache.spark.mllib.recommendation.cofirank

import org.apache.spark.{SparkContext, SparkConf}

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
    model.save(sc, "hdfs://node-6:9000/testdata")
  }
}
```

##Reference
[1]Weimer M, Karatzoglou A, Le Q V, et al. CoFiRank - Maximum Margin Matrix Factorization for Collaborative Ranking[J]. 2007, 20:1593-1600.

[2]Dai Y H, Fletcher R. New algorithms for singly linearly constrained quadratic programs subject to lower and upper bounds[J]. Mathematical Programming, 2006, 106(3):403-421.
