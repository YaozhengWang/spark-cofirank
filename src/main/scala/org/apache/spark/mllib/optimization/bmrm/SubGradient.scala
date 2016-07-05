package org.apache.spark.mllib.optimization.bmrm

import java.security.InvalidAlgorithmParameterException

import breeze.numerics.{log2}
import org.apache.spark.annotation.DeveloperApi
import org.apache.spark.mllib.linalg.{Vectors, Vector}
import scala.math.pow
import org.apache.spark.mllib.linalg.BLAS.{dot, copy}
import scala.collection.mutable

/**
 * Created by cage on 2016/6/1.
 */
/**
 * :: DeveloperApi::
 * Class used to compute the gradient for a loss function, given a set of data points.
 */
@DeveloperApi
abstract class SubGradient extends Serializable{
  /**
   * Compute the gradient and loss given the features of a set of date points.
   *
   * @param data features for a set of data points
   * @param label labels for these data points
   * @param weights weights/coefficients corresponding to features
   *
   * @return (gradient: Vector, loss: Double)
   */
  def compute(data: Array[Vector], label: Array[Double], weights: Vector): (Vector, Double) = {
    val gradient = Vectors.zeros(weights.size)
    val trainK = label.length
    val c_exponent = -0.25
    val loss = compute(data, label, weights, gradient, trainK, c_exponent)
    (gradient, loss)
  }

  /**
   * Compute the gradient and loss given the features of a set of data points, add the
   * gradient to a provided vector to avoid creating new objects, and return loss.
   *
   * @param data features for a set of data points
   * @param label labels for these data points
   * @param weights weights/coefficients corresponding to features
   * @param comGradient the computed gradient will be added to this vector
   * @param trainK the truncation threshold k of ndcg
   * @param c_exponent the coefficients of c
   *
   * @return loss
   */
  def compute(data: Array[Vector], label: Array[Double], weights: Vector, comGradient: Vector, trainK: Int, c_exponent: Double): Double
}

/**
 * :: DeveloperApi::
 * Compute gradient and loss for l(f, y) := max_{pi}[delta(pi, y) + <c, f_{pi}-f>], which is
 * used in Cofirank
 */
@DeveloperApi
class NdcgSubGradient extends SubGradient {

  override def compute(data: Array[Vector], label: Array[Double], weights: Vector): (Vector, Double) = {
    /** Init the truncation threshold k, Default: label.length*/
    val trainK = data.length
    /** Default: c_exponent: -0.25 */
    val c_exponent = -0.25
    val gradient = Vectors.zeros(weights.size)
    val loss = compute(data, label, weights, gradient, trainK, c_exponent)
    (gradient, loss)
  }

  override def compute(data: Array[Vector], label: Array[Double],
       weights: Vector, comGradient: Vector, trainK: Int, c_exponent: Double): Double = {
    /** Init the variable c*/
    val c : Array[Double] = (0 until label.length).toArray[Int].map(i => pow((i+1.0), c_exponent))
    /** Compute predictive value: f  */
    val f = (0 until data.length).toArray[Int].map(i=> dot(data(i), weights))
    /** Compute the decreasing sort of label(y) */
    val dsBuilder = mutable.ArrayBuilder.make[Int]
    label.zipWithIndex.sorted.reverse.foreach(x => dsBuilder += x._2)
    val decreasingSort  = dsBuilder.result()
    /** Compute the perfectDCG of the decreasing sort */
    val perfectDCG = dcg(label, decreasingSort, trainK)
    //val pi: Array[Int] = Array.fill(label.length)(0)
    /** Compute pi to make l(f, y) maximize */
    val pi = find_permutation(label, f, trainK, perfectDCG, c)
    /** Inverse permutation */
    val pii = pi.reverse

    var scalarProd = 0.0
    val grad = Array.fill(label.length)(0.0)
    for (i <- 0 until label.length) {
      scalarProd += c(i) * (f(pi(i)) - f(i))
      grad(i) = c(pii(i)) - c(i)
    }
    /** Compute the subGradient of l(f, y) */
    val comGrad = Array.fill(weights.size)(0.0)
    for (i <- 0 until data(0).size) {
      for (j <- 0 until data.length) {
        comGrad(i) += data(j)(i) * grad(j)
      }
    }
    /*
    val valueSet = mutable.ArrayBuilder.make[Double]
    val indexSet = mutable.ArrayBuilder.make[Int]
    comGrad.zipWithIndex.foreach{x =>
      valueSet += x._1
      indexSet += x._2
    }
    */
    copy(Vectors.dense(comGrad), comGradient)

    /** Compute loss function*/
    delta(label, pi, trainK, perfectDCG) + scalarProd
  }

  /** Transpose the Matrix */
  /*
  def transposeDouble(xss: Array[Array[Double]]): Array[Array[Double]] = {
    for (i <- Array.range(0, xss(0).length)) yield
      for (xs <- xss) yield xs(i)
  }
  */
  /** Compute the min number of element*/
  /*
  def minElemNumber(xs: Array[Array[Double]]): Int = {
    var count = 0
    for (x <- xs) {
      val minElem = x.min
      for (elem <- x){
        if (elem == minElem)
          count += 1
      }
    }
    count
  }
  */

  /** Each row of the matrix minus the smallest element of this row */
  /*
  def minusMinElem(xs: Array[Array[Double]]): Array[Array[Double]] = {
    for (i <- 0 until xs.length) {
      val minValue = xs(i).min
      for (j <- 0 until xs(i).length) {
        xs(i)(j) -= minValue
      }
    }
    xs
  }
  */
  /**  Assert only one zero in an array */
  /*
  def isOneZero(xs: Array[Double]): (Boolean, Int) = {
    var count = 0
    var index = 0
    for (i <- 0 until xs.length) {
      if (xs(i) == 0){
        count += 1
        index = i
      }
    }
    (count == 1, index)
  }
  */

  /** Compute the permutation by linear assignment and store in pi */
  def find_permutation(label: Array[Double], f: Array[Double],
      trainK: Int, perfectDCG: Double, c: Array[Double]): Array[Int] = {
    /** setting up C_ij */
    val C = Array.fill(label.length){
      Array.fill(label.length)(0.0)
    }
    for (i <- 0 until label.length) {
      for (j <- 0 until label.length) {
        if (i < trainK) {
          C(i)(j) = ((pow(2, label(j)) - 1) / log2(i + 2)) / perfectDCG - c(i) * f(j)
        } else {
          C(i)(j) = -c(i) * f(j)
        }
      }
    }

    /** initialize the row and col */
    val col = Array.fill(label.length)(0)
    val row = Array.fill(label.length)(0)

    /** create and initialize parent_row and unchosen_row */
    val parent_row = Array.fill(label.length)(0)
    val unchosen_row = Array.fill(label.length)(0)

    /** create and initialize row_dec, col_inc and slack */
    val row_dec = Array.fill(label.length)(0.0)
    val col_inc = Array.fill(label.length)(0.0)
    val slack = Array.fill(label.length)(0.0)

    /** create and initialize slack_row */
    val slack_row = Array.fill(label.length)(0)
    var s = 0.0
    for (l <- 0 until label.length){
      s = C(0)(l)
      for (k <- 1 until label.length)
        if (C(k)(l) < s)
          s = C(k)(l)
      if (s != 0)
        for (k <- 0 until label.length)
          C(k)(l) -= s
    }
    var t = 0
    for (l <- 0 until label.length){
      row(l) = -1
      parent_row(l) = -1
      col_inc(l) = 0
      slack(l) = 999999.9
    }
    var flag3 = 0
    for (k <- 0 until label.length){
      flag3 = 0
      s = C(k)(0)
      for (l <- 1 until label.length)
        if (C(k)(l) < s)
          s = C(k)(l)
      row_dec(k) = s
      var l = 0
      while (flag3 == 0 && l < label.length) {
        if (s == C(k)(l) && row(l) < 0) {
          col(k) = l
          row(l) = k
          flag3 = 1
        }
        l += 1
      }
      if (flag3 != 1) {
        col(k) = -1
        unchosen_row(t) = k
        t += 1
      }
    }
    if (t == 0)
     return col
    var unmatched = t
    var q = 0
    var k = 0
    var total = 0.0
    var j = 0
    var flag = 0
    var flag2 = 0
    while (true) {
      q = 0
      flag = 0
      flag2 = 0
      while(flag != 1 && flag2 != 1) {
        while (flag != 1 && q < t) {
          k = unchosen_row(q)
          s = row_dec(k)
          var l = 0
          while (flag != 1 && l < label.length) {
            if (slack(l) != 0) {
              total = C(k)(l)-s+col_inc(l)
              if (total < slack(l)){
                if (total == 0) {
                  if (row(l) < 0) {
                    flag = 1
                    var flag4 = 0
                    while(flag4 != 1) {
                      j = col(k)
                      col(k) = l
                      row(l) = k
                      if (j < 0)
                        flag4 = 1
                      if (flag4 == 0) {
                        k = parent_row(j)
                        l = j
                      }
                    }
                    unmatched -= 1
                    if (unmatched == 0)
                      return col
                    t = 0
                    for (l <- 0 until label.length){
                      parent_row(l) = -1
                      slack(l) = 999999.9
                    }
                    for (k <- 0 until(label.length))
                      if (col(k) < 0){
                        unchosen_row(t) = k
                        t += 1
                      }
                  }
                  if (flag == 0) {
                    slack(l) = 0
                    parent_row(l) = k
                    unchosen_row(t) = row(l)
                    t += 1
                  }
                }else {
                  slack(l) = total
                  slack_row(l) = k
                }
              }
            }
            if (flag == 0) {
              l += 1
            }
          }
          if (flag == 0) {
            q += 1
          }
        }
        if (flag == 0) {
          s = 999999.9
          for (l <- 0 until label.length)
            if (slack(l) != 0 && slack(l) < s)
              s = slack(l)
          for (q <- 0 until t)
            row_dec(unchosen_row(q)) += s
          var l = 0
          while (flag2 != 1 && l < label.length) {
            if (slack(l) != 0) {
              slack(l) -= s
              if (slack(l) == 0) {
                k = slack_row(l)
                if (row(l) < 0) {
                  j = l + 1
                  while (j < label.length) {
                    if (slack(j) == 0)
                      col_inc(j) += s
                    j += 1
                  }
                  flag2 = 1
                  var flag5 = 0
                  while (flag5 != 1) {
                    j = col(k)
                    col(k) = l
                    row(l) = k
                    if (j < 0)
                      flag5 = 1
                    if (flag5 == 0) {
                      k = parent_row(j)
                      l = j
                    }
                  }
                  unmatched -= 1
                  if (unmatched == 0)
                    return col
                  t = 0
                  for (l <- 0 until label.length) {
                    parent_row(l) = -1
                    slack(l) = 999999.9
                  }
                  for (k <- 0 until (label.length))
                    if (col(k) < 0) {
                      unchosen_row(t) = k
                      t += 1
                    }
                } else {
                  parent_row(l) = k
                  unchosen_row(t) = row(l)
                  t += 1
                }
              }
            } else col_inc(l) += s
            if (flag2 == 0) {
              l += 1
            }
          }
        }
      }
    }
    /** transpose matrix C */
    //val tran_C = transposeDouble(C)
    /** The implement of the KuhnMunkres algorithm*/
    /** The number of min element in rows */
   // val rows = minElemNumber(C)
    /** The number of min element in cols */
   // val cols = minElemNumber(tran_C)
    /*
    var C_final:Array[Array[Double]] = Array.fill(label.length) {
      Array.fill(label.length)(0.0)
    }
    */
    /*
    if (rows > cols){
      minusMinElem(tran_C)
      C_final = transposeDouble(tran_C)
      minusMinElem(C_final)
    } else {
      minusMinElem(C)
      val tran_C_final = transposeDouble(C)
      minusMinElem(tran_C_final)
      C_final = transposeDouble(tran_C_final)
    }
    */


    /** Find the independent elements of zeros */
    /*
    val row = Array.fill(label.length)(-1)
    val col = Array.fill(label.length)(-1)
    val pi = Array.fill(label.length)(-1)
    while(pi.contains(-1)) {
      for (i <- 0 until label.length) {
        if (row(i) == -1 && isOneZero(C_final(i))._1) {
          row(i) = 1
          col(isOneZero(C_final(i))._2) = 1
          pi(i) = isOneZero(C_final(i))._2
          for (x <- C_final)
            x(isOneZero(C_final(i))._2) = 1
        }
      }
      for (j <- 0 until label.length) {
        if (col == -1){
          val colBuilder = mutable.ArrayBuilder.make[Double]
          for (x <- C_final)
            colBuilder += x(j)
          if (isOneZero(colBuilder.result())._1){
            row(isOneZero(colBuilder.result())._2) = 1
            col(j) = 1
            pi(isOneZero(colBuilder.result())._2) = j
            C_final(isOneZero(colBuilder.result())._2) = Array.fill(label.length)(1)
          }
        }
      }
    }
    */
    col
  }

  /** Compute the loss and store in value */
  def delta(label: Array[Double], pi: Array[Int], trainK: Int, perfectDCG: Double): Double = {
    /** Based on the sort of pi and compute the DCG of the sort pi */
    val theDCG = dcg(label, pi, trainK)

    /** Compute the NDCG@k*/
    val nDCG = theDCG / perfectDCG
    val result = 1.0 - nDCG
    assert(result >= 0.0 && result <= 1.0)
    result
  }

  /** Compute the dcg of label(y) by sort pi */
  def dcg(label: Array[Double], pi: Array[Int], trainK: Int): Double = {
    if (label.length < trainK){
      throw new InvalidAlgorithmParameterException("trainK is bigger than the length of label.")
    }
    if (pi.length < trainK) {
      throw new InvalidAlgorithmParameterException("The given permutation vector pi is too small.")
    }
    if (pi.length != label.length) {
      throw new InvalidAlgorithmParameterException("The given permutation vector pi has a different size than the vector label")
    }

    var d = 0.0
    (0 until trainK).map{i =>
      d += (pow(2.0, label(pi(i)))-1.0) / log2(i+2.0)
    }
    d
  }
}
/*
object NdcgSubGradient {
  def generateSubInput(nPoint: Int, dim: Int,seed: Int):(Array[Vector], Array[Double], Vector) = {
    val rnd = new Random(seed)
    val label = Array.fill[Double](nPoint)(rnd.nextInt(5)+1.0)
    val testData = Array.fill[Vector](nPoint)(Vectors.dense(Array.fill(dim)(rnd.nextInt(10)+1.0)))
    val initWeights = Vectors.dense(Array.fill(dim)(rnd.nextInt(10)+1.0))
    (testData, label, initWeights)
  }
  def main(args: Array[String]) = {
  //  val conf = new SparkConf().setAppName("BMRM").setMaster("spark://node-6:7077").setJars(List("D:\\IdeaProjects\\spark-cofirank\\out\\artifacts\\spark_cofirank_jar\\spark-cofirank.jar"))
  //  val sc = new SparkContext(conf)
    val subGrad = new NdcgSubGradient()
    val (testData, label, initWeights) = generateSubInput(500, 10, 45)
    val (gradient, loss) = subGrad.compute(testData, label, initWeights)
    println(gradient)
    println(loss)
  }
}
*/
