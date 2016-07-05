package org.apache.spark.mllib.optimization.bmrm

import breeze.linalg.min
import org.apache.spark.annotation.DeveloperApi
import org.apache.spark.mllib.linalg.{Vectors, Vector}
import org.apache.spark.mllib.linalg.BLAS.{dot, copy}

/**
 * Created by cage on 2016/6/5.
 */
class BMRM (private var gradient: SubGradient, private var updater: SubUpdater){
  private var trainK: Int = 2
  private var c_exponent: Double = -0.25
  private var lambda: Double = 1.0
  private var epsilonTol = 0.1
  private var maxNumOfIter = 100

  /**
   * trainK is a truncation threshold of ndcg
   */
  def setTrainK(trainK: Int): this.type = {
    this.trainK = trainK
    this
  }

  /**
   * c exponent for NDCG loss
   */
  def setC_exponent(c_exponent: Double): this.type = {
    this.c_exponent = c_exponent
    this
  }

  /**
   * the coefficient of updater
   */
  def setLambda(lambda: Double): this.type = {
    this.lambda = lambda
    this
  }

  /**
   * Tolerance for epsilon termination criterion.
   * epsilon = min_{t' <= t}J(w_{t'}) - J_t(w_t)
   * where J_t is piece-wise linear approx of J
   */
  def setEpsilonTol(epsilonTol: Double): this.type = {
    require(0.0 <= epsilonTol && epsilonTol <= 1.0)
    this.epsilonTol = epsilonTol
    this
  }

  /**
   * Maximum number of BMRM iteration
   */
  def setMaxNumOfIter(maxNumOfIter: Int): this.type = {
    this.maxNumOfIter = maxNumOfIter
    this
  }

  /**
   * Set the gradient function(of the loss function of a set of data examples)
   * to be used
   */
  def setGradient(gradient: SubGradient): this.type = {
    this.gradient = gradient
    this
  }

  /**
   * Set the updater function to actually perform a step in a given direction.
   */
  def setUpdater(updater: SubUpdater): this.type = {
    this.updater = updater
    this
  }

  /**
   * :: DeveloperApi ::
   * @param data  the features of train set
   * @param label the labels of train set
   * @param initialWeights  initial weights
   * @return solution vector
   */
  @DeveloperApi
  def optimize(data: Array[Vector], label: Array[Double], initialWeights: Vector): Vector = {
    val (weights, _) = BMRM.train(data, label, gradient, updater, maxNumOfIter, trainK, c_exponent, lambda, initialWeights, epsilonTol)
    weights
  }
}

/**
 *  :: DeveloperApi ::
 *  Top-level method to run bmrm
 */
@DeveloperApi
object BMRM {
  /**
   * Run a BMRM algorithm
   * @param data  Input data for bmrm
   * @param label label data for bmrm
   * @param gradient  Gradient object (used to compute the gradient of the loss function of a set of data examples)
   * @param updater Updater function to actually perform a gradient step in a given direction.
   * @param trainK truncation threshold of ndcg
   * @param c_exponent c exponent for NDCG loss
   * @param lambda  the coefficient of updater
   * @param initialWeights initial weights
   * @param epsilonTol  Tolerance for epsilon termination criterion.
   * @return  weight final
   */
  def train(
      data: Array[Vector],
      label: Array[Double],
      gradient: SubGradient,
      updater: SubUpdater,
      maxNumOfIter: Int,
      trainK: Int,
      c_exponent: Double,
      lambda: Double,
      initialWeights: Vector,
      epsilonTol: Double
    ): (Vector, Double) = {
    var loss = 0.0
    var minExactObjVal = 100000.0
    var finalExactObjVal = 0.0
    var finalLoss = 0.0
    var finalRegVal = 0.0
    var iter = 0
    var epsilon = 0.0
    var approxObjVal = 100000.0

    val grad = Vectors.zeros(initialWeights.size)
    val w_final = Vectors.zeros(initialWeights.size)
    var flag1 = 0
    var flag2 = 0


    require(epsilonTol > 0.0 && epsilonTol <1.0)
    while (flag1 != 1 && flag2 != 1) {
      iter += 1
      /** Compute sub gradient and loss function. */
      loss = gradient.compute(data, label, initialWeights, grad, trainK, c_exponent)
      /** Value of the regularize term */
      val regVal =  computeRegularizeValue(initialWeights)
      val exactObjVal = loss + regVal
      minExactObjVal = min(minExactObjVal, exactObjVal)

      if (iter == 2) {
        finalExactObjVal = exactObjVal
        finalLoss = loss
        finalRegVal = regVal
        copy(initialWeights, w_final)
      } else if (iter > 2) {
        if (finalExactObjVal > exactObjVal) {
          finalExactObjVal = exactObjVal
          finalLoss = loss
          finalRegVal = regVal
          copy(initialWeights, w_final)
        }
      }
      epsilon = fabs(minExactObjVal - approxObjVal)
      println(epsilon)

      if (iter >= maxNumOfIter) {
        println("epsilon criterion:" + epsilon)
        flag1 = 1
      }

      if (iter >= 2){
        if (epsilon < epsilonTol){
          println("epsilon criterion:" + epsilon)
          flag2 = 1
        }
      }

      /** Compute the updater */
      if (flag1 == 0 && flag2 == 0) {
        val upvalue = updater.compute(initialWeights, grad, loss, lambda)
        copy(upvalue._1, initialWeights)
        approxObjVal = upvalue._2
      }
    }
    copy(w_final, initialWeights)
    (initialWeights, loss)
  }

  /**
   * Compute the regularize term
   * @param w weights
   * @return regularize value
   */
  def computeRegularizeValue(w: Vector): Double = 0.5 * dot(w, w)

  /** Auxiliary function: solve for absolute value */
  def fabs(x: Double) = if (x < 0) -x else x
}

