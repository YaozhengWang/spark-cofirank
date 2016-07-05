package org.apache.spark.mllib.util

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.SQLContext
import org.scalatest.{Suite, BeforeAndAfterAll}

/**
 * Created by cage on 2016/6/19.
 */

/**
 * Testing helper excerpted from the spark testing library.
 * @see [[https://github.com/apache/spark/blob/master/mllib/src/test/scala/org/apache/spark/mllib/util/MLlibTestSparkContext.scala]]
 */
trait MLlibTestSparkContext extends BeforeAndAfterAll { self: Suite =>
  @transient var sc: SparkContext = _
  @transient var sqlContext: SQLContext = _

  override def beforeAll(): Unit = {
    super.beforeAll()
    val conf = new SparkConf().setMaster("local[2]").setAppName("MLlibUnitTest")
    sc = new SparkContext(conf)
    sc.setLogLevel("WARN")
    sqlContext = new SQLContext(sc)
  }

  override def afterAll(): Unit = {
    sqlContext = null
    if (sc != null) {
      sc.stop()
    }
    sc = null
    super.afterAll()
  }
}
