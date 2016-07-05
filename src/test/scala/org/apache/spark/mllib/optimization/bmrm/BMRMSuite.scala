package org.apache.spark.mllib.optimization.bmrm

import org.scalatest.FunSuite
import org.apache.spark.mllib.util.MLlibTestSparkContext
import org.apache.spark.mllib.linalg.{Vectors, Vector}

import scala.util.Random

/**
 * Created by cage on 2016/6/19.
 */

object BMRMSuite {
  def generateSubInput(nPoint: Int, dim: Int,seed: Int):(Array[Vector], Array[Double], Vector) = {
    val rnd = new Random(seed)
    val label = Array.fill[Double](nPoint)(rnd.nextInt(5)+1.0)
    val testData = Array.fill[Vector](nPoint)(Vectors.dense(Array.fill(dim)(rnd.nextInt(10)+1.0)))
    val initWeights = Vectors.dense(Array.fill(dim)(rnd.nextInt(10)+1.0))
    (testData, label, initWeights)
  }
}

class BMRMSuite extends FunSuite with MLlibTestSparkContext {

  test("Test the loss and gradient of first iteration") {
    val subGrad = new NdcgSubGradient()
    val (testData, label, initWeights) = BMRMSuite.generateSubInput(100, 100, 45)
    val (gradient, loss) = subGrad.compute(testData, label, initWeights)
    println(gradient)
    println(loss)
  }

  test("Test the update of the weights of first iteration") {
    val subGrad = new NdcgSubGradient()
    val (testData, label, initWeights) = BMRMSuite.generateSubInput(100, 1000, 45)
    val (gradient, loss) = subGrad.compute(testData, label, initWeights)
    val subUpdater = new DaiFletcherUpdater()
    val (newWeights, objval) = subUpdater.compute(initWeights, gradient, loss, 1.0)
    println(initWeights)
    println(loss)
    println(newWeights)
    println(objval)
  }

  test("Test the BMRM optimization") {
    val subGrad = new NdcgSubGradient()
    val subUpdater = new DaiFletcherUpdater()
    val bmrm = new BMRM(subGrad, subUpdater)
    val (testData, label, initWeights) = BMRMSuite.generateSubInput(100, 10, 45)
    println(initWeights)
    val newWeights = bmrm.optimize(testData, label, initWeights)
    println(newWeights)
  }
}