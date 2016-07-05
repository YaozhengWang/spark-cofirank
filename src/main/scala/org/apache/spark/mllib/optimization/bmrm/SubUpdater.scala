package org.apache.spark.mllib.optimization.bmrm

import breeze.numerics.sqrt

import scala.collection.mutable
import org.apache.spark.annotation.DeveloperApi
import org.apache.spark.mllib.linalg.{Vectors, Vector}
import org.apache.spark.mllib.linalg.BLAS.dot

import scala.collection.mutable.ArrayBuffer

/**
 * Created by cage on 2016/6/5.
 */
/**
 * :: DeveloperApi ::
 * Class used to perform steps (weight update) using DaiFletcherPGM
 *
 * For general minimization problems, or for regularized problems of the form
 *                  min L(w) + regParam * R(w)
 * the compute function performs the actual update step, when DaiFletcherPGM is used.
 *
 * The SubUpdater is responsible to also perform the update coming from the regularization
 * term R(w)(if any regularization is used).
 */
@DeveloperApi
abstract class SubUpdater extends Serializable{
  /**
   * Compute an updated value for weights given the gradient, loss, approxObjVal and
   * regularization parameter.
   *
   * @param weightsOld weight/coefficients for features
   * @param gradient sub gradient for R(w_{t-1})
   * @param loss loss function for R(w_{t-1})
   * @param regParam Regularization parameter
   *
   * @return the updated weights
   */
  def compute(
    weightsOld: Vector,
    gradient: Vector,
    loss: Double,
    regParam: Double): (Vector, Double)
}

/**
 * :: DeveloperApi::
 * A DaiFletcher Updater for Cofirank with L2 regularized problems.
 *             R(w) = 1/2 ||w||2
 */
@DeveloperApi
class DaiFletcherUpdater extends SubUpdater {
  /** Gradient set */
  private val gradientSet = mutable.ArrayBuilder.make[Vector]
  /** Offsets set */
  private val offsetSet = mutable.ArrayBuilder.make[Double]
  /** Hessian matrix of the objective function */
  private val Q = mutable.ArrayBuilder.make[mutable.ArrayBuffer[Double]]
  /** The variables */
  private val x = ArrayBuffer[Double]()
  /** Linear part of the objective function */
  private val f = mutable.ArrayBuilder.make[Double]


  override def compute(
    weightsOld: Vector,
    gradient: Vector,
    loss: Double,
    regParam: Double): (Vector, Double) = {
    /** Compute the slope of the tangent */
    val w_dot_grad =dot(weightsOld, gradient)
    Update(gradient, loss-w_dot_grad)
    SolveQP()
    GetSolution(regParam)
  }

  /**
   * Return solution (w, objval)
   * @param lambda  lambda is regularization constant
   * @return (w, objval)
   */
  def GetSolution(lambda: Double) :(Vector, Double) = {
    val factor = 1.0 / lambda
    var objval = 0.0
    val w = mutable.ArrayBuilder.make[Double]
    /** Compute objective value */
    val temp = Array.fill(x.length)(0.0)
    /*
    val valueSet = mutable.ArrayBuilder.make[Double]
    val indexSet = mutable.ArrayBuilder.make[Int]
    x.result().zipWithIndex.foreach{x =>
      valueSet += x._1
      indexSet += x._2
    }
    */
    val xv = Vectors.dense(x.toArray)
    for (i <- 0 until(Q.result().length)){
      /** Convert Q.result()(i) into Spark Vector */
      /*
      val valueSetQ = mutable.ArrayBuilder.make[Double]
      val indexSetQ = mutable.ArrayBuilder.make[Int]
      Q.result()(i).zipWithIndex.foreach{x =>
        valueSetQ += x._1
        indexSetQ += x._2
      }
      */
      val Q_i = Vectors.dense(Q.result()(i).toArray)
      temp(i) = dot(Q_i, xv)
    }
    for (i <- 0 until x.length){
      objval += x(i) * (-0.5 * factor * temp(i) - f.result()(i))
    }

    /** Compute w which is updated */
    for (i <- 0 until gradientSet.result()(0).toArray.length){
      var value = 0.0
      for (j <- 0 until x.length)
        value += x(j) * gradientSet.result()(j)(i)
      w += -factor * value
    }
    /*
    val valueSetw = mutable.ArrayBuilder.make[Double]
    val indexSetw = mutable.ArrayBuilder.make[Int]
    w.result().zipWithIndex.foreach{x =>
      valueSetw += x._1
      indexSetw += x._2
    }
    */
    val wv = Vectors.dense(w.result())
    (wv, objval)
  }

  /**
   * Initialization
   * for k = 1, 2, ... until converged
   *    Calculate the projection step using Algorithm 1,
   *    Possibly carry out a line search,
   *    Calculate a BB-like step length,
   *    Update the line search control parameters
   * end
   */
  def SolveQP() = {
    var akold, bkold, max, alpha, gd, ak, bk, lamnew = 0.0
    /** default values */
    val maxPGMIter = 3000
    val tol = 1e-6
    /** variables for the adaptive nonmonotone linesearch */
    var L, llast = 0
    var fr, fbest, fv, fc, fv0 = 0.0
    /** Define gradient vector */
    val tempv = Array.fill(x.length)(0.0)
    val g = Array.fill(x.length)(0.0)
    val y = Array.fill(x.length)(0.0)
    val t = Array.fill(x.length)(0.0)
    val d = Array.fill(x.length)(0.0)
    val Qd = Array.fill(x.length)(0.0)
    val xplus = Array.fill(x.length)(0.0)
    val tplus = Array.fill(x.length)(0.0)
    val sk = Array.fill(x.length)(0.0)
    val yk = Array.fill(x.length)(0.0)
    for (i <- 0 until(x.length))
      tempv(i) = -x(i)
    /** Find the starting point of the algorithm and map it into the feasible region. */
    val x_k = ProjectDF(x.length, Array.fill(x.length)(1.0),
      1.0, tempv, Array.fill(x.length)(0.0), Array.fill(x.length)(1.0), x.toArray)
    /** Gradient vector: g = Q*x_k + f */
    /** Convert x_k into Spark Vector */
    /*
    val valueSet = mutable.ArrayBuilder.make[Double]
    val indexSet = mutable.ArrayBuilder.make[Int]
    x_k.zipWithIndex.foreach{x =>
      valueSet += x._1
      indexSet += x._2
    }
    */
    val x_kv = Vectors.dense(x_k)
    for (i <- 0 until(Q.result().length)){
      /** Convert Q.result()(i) into Spark Vector */
      /*
      val valueSetQ = mutable.ArrayBuilder.make[Double]
      val indexSetQ = mutable.ArrayBuilder.make[Int]
      Q.result()(i).zipWithIndex.foreach{x =>
        valueSetQ += x._1
        indexSetQ += x._2
      }
      */
      val Q_i = Vectors.dense(Q.result()(i).toArray)
      t(i) = dot(x_kv, Q_i)
      g(i) = dot(x_kv, Q_i) + f.result()(i)
      y(i) = g(i) - x_k(i)
    }
    /** Find the next point of the algorithm and map it into the feasible region. */
    val x_kplus = ProjectDF(x.length, Array.fill(x.length)(1.0),
      1.0, y, Array.fill(x.length)(0.0), Array.fill(x.length)(1.0), tempv)

    /** Compute y */
    max = 1e-10
    for (i <- 0 until x.length) {
      y(i) = x_kplus(i) - x_k(i)
      if (fabs(y(i)) > max)
        max = fabs(y(i))
    }
    alpha = 1.0 / max
    /** Compute loss Function value */
    for (i <- 0 until x.length)
      fv0 += x_k(i) * (0.5 * t(i) + f.result()(i))
    L = 2
    fr = 1e10
    fbest = fv0
    fc = fv0
    var flag = 0
    var innerIter = 1
    while (flag != 1 && innerIter < maxPGMIter) {
      /** The first stage in the loop of Algorithm 2 is to take a steepest descent step
        * from a current iterate x_k with fixed step length alpha
        */
      for (i <- 0 until x.length)
        tempv(i) = alpha * g(i) - x_k(i)

      val y_k = ProjectDF(x.length, Array.fill(x.length)(1.0),
        1.0, tempv, Array.fill(x.length)(0.0), Array.fill(x.length)(1.0), y)

      /** d is a feasible descent direction */
      for (i <- 0 until(x.length)){
        d(i) = y_k(i) - x_k(i)
        gd += d(i) * g(i)
      }

      /**
       * The second stage of Algorithm 2 is to decide if a line search is necessary.
       */
      /** Firstly, compute Qd = Q*d */
      /** Convert d into Spark Vector */
      /*
      val valueSetd = mutable.ArrayBuilder.make[Double]
      val indexSetd = mutable.ArrayBuilder.make[Int]
      d.zipWithIndex.foreach{x =>
        valueSetd += x._1
        indexSetd += x._2
      }
      */
      val dv = Vectors.dense(d)
      for (i <- 0 until(Q.result().length)){
        /** Convert Q.result()(i) into Spark Vector */
        /*
        val valueSetQ = mutable.ArrayBuilder.make[Double]
        val indexSetQ = mutable.ArrayBuilder.make[Int]
        Q.result()(i).zipWithIndex.foreach{x =>
          valueSetQ += x._1
          indexSetQ += x._2
        }
        */
        val Q_i = Vectors.dense(Q.result()(i).toArray)
        Qd(i) = dot(Q_i, dv)
      }
      ak = 0.0
      bk = 0.0
      for (i <- 0 until x.length) {
        ak += d(i) * d(i)
        bk += d(i) * Qd(i)
      }
      if (bk > 1e-20*ak && gd < 0.0)
        lamnew = - gd / bk
      else
        lamnew = 1.0

      fv = 0.0
      for (i <- 0 until(x.length)){
        xplus(i) = x_k(i) + d(i)
        tplus(i) = t(i) + Qd(i)
        /** Loss function value */
        fv += xplus(i) * (0.5 * tplus(i) + f.result()(i))
      }

      /** When a line search is necessary. */
      if ((innerIter == 1 && fv >= fv0) || (innerIter > 1 && fv >= fr)){
        fv = 0.0
        for (i <- 0 until x.length){
          xplus(i) = x_k(i) + lamnew * d(i)
          tplus(i) = t(i) + lamnew * Qd(i)
          fv += xplus(i) * (0.5 * tplus(i) + f.result()(i))
        }
      }
      /** update s_k, y_k */
      for (i <- 0 until(x.length)){
        sk(i) = xplus(i) - x_k(i)
        yk(i) = tplus(i) - t(i)
        x_k(i) = xplus(i)
        t(i) = tplus(i)
        g(i) = t(i) + f.result()(i)
      }
      /** Update the line search control parameters */
      if (fv < fbest){
        fbest = fv
        fc = fv
        llast = 0
      } else {
        if (fv > fc)
          fc = fv
        llast += 1
        if (llast == L){
          fr = fc
          fc = fv
          llast = 0
        }
      }

      /** update step length */
      ak = 0.0
      bk = 0.0
      for (i <- 0 until x.length){
        ak += sk(i) * sk(i)
        bk += sk(i) * yk(i)
      }
      if (bk <= 1e-20 * ak)
        alpha = 1e10
      else {
        if (bkold < 1e-20 * akold)
          alpha = ak / bk
        else
          alpha = (akold + ak) / (bkold + bk)

        if (alpha > 1e10)
          alpha = 1e10
        else if (alpha < 1e-10)
          alpha = 1e-10
      }
      akold = ak
      bkold = bk

      /**
       * stopping criterion based on tol
       */

      bk = 0.0
      for (i <- 0 until x.length)
        bk += x_k(i) * x_k(i)

      if (sqrt(ak) < tol*0.5*sqrt(bk))
        flag = 1

      innerIter += 1
    }
    /** Given x_k to x */
    for (i <- 0 until x.length)
      x(i) = x_k(i)
  }

  /**
   * Piecewise linear monotone target function for the Dai-Fletcher projector.
   */
  def ProjectR(x: Array[Double], n: Int, lambda: Double,
    a: Array[Double], b: Double, c: Array[Double],
    l: Array[Double], u: Array[Double]): Double = {
    var r = 0.0

    for (i <- 0 until(n)){
      x(i) = -c(i) + lambda*a(i)
      if (x(i) > u(i))
        x(i) = u(i)
      else if (x(i) < l(i))
        x(i) = l(i)
      r += a(i)*x(i)
    }
    r - b
  }

  /**
   * Modified Dai-Fletcher QP projector solves the problem:
   *
   *      minimise 0.5*x'*x + c'*x
   *      subj to a'*x <= b
   *            l <= x <= u
   * @param n The dimension of a,c,l,u,x
   * @param a As shown in the above equation
   * @param b A scalar
   * @param c The point to be projected onto feasible set
   * @param l The lower bound of x
   * @param u The upper bound of x
   * @param x which needs to be solved
   */
  def ProjectDF(n: Int, a: Array[Double], b: Double,
    c: Array[Double], l: Array[Double], u: Array[Double],
    x: Array[Double]):Array[Double] = {
    var lambda, lambdal, lambdau, lambda_new = 0.0
    var dlambda = 0.5
    var r, rl, ru, s = 0.0
    val tol_r = 1e-15
    val tol_lam = 1e-15

    /** Bracketing Phase */
    /** Calculate x by x(lambda) = mid(l, h, u); r = aTx -b */
    r = ProjectR(x, n, lambda, a, b, c, l, u)

    if (fabs(r) < tol_r)
      return x
    /** If r < 0, lambda search for positive direction */
    if (r < 0.0){
      lambdal = lambda
      rl = r
      lambda = lambda + dlambda
      r = ProjectR(x, n, lambda, a, b, c, l, u)
      while (r < 0.0) {
        lambdal = lambda
        s = rl/r - 1.0
        if (s < 0.1) s = 0.1
        dlambda = dlambda + dlambda/s
        lambda = lambda + dlambda
        rl = r
        r = ProjectR(x, n, lambda, a, b, c, l, u)
      }
      lambdau = lambda
      ru = r
    }
    else {
      /** If r > 0, lambda search for negative direction */
      lambdau = lambda
      ru = r
      lambda = lambda - dlambda
      r = ProjectR(x, n, lambda, a, b, c, l, u)
      while (r > 0.0) {
        lambdau = lambda
        s = ru/r - 1.0
        if (s < 0.1) s = 0.1
        dlambda = dlambda + dlambda/s
        lambda = lambda - dlambda
        ru = r
        r = ProjectR(x, n, lambda, a, b, c, l, u)
      }
      lambdal = lambda
      rl = r
    }

    /** Secant Phase */
    s = 1.0 - rl/ru
    dlambda = dlambda/s
    lambda = lambdau - dlambda
    r = ProjectR(x, n, lambda, a, b, c, l, u)
    /** while not converged */
    while (fabs(r) > tol_r && dlambda > tol_lam*(1.0 + fabs(lambda))){
      if (r > 0.0){
        if (s <= 2.0){
          lambdau = lambda
          ru = r
          s = 1.0 - rl/ru
          dlambda = (lambdau - lambdal)/s
          lambda = lambdau - dlambda
        } else {
          s = ru/r - 1.0
          if (s < 0.1) s = 0.1
          dlambda = (lambdau -lambda)/s
          lambda_new = 0.75*lambdal + 0.25*lambda
          if (lambda_new < (lambda - dlambda))
            lambda_new = lambda - dlambda
          lambdau = lambda
          ru = r
          lambda = lambda_new
          s = (lambdau - lambdal) / (lambdau - lambda)
        }
      }
      else {
        if (s >= 2.0){
          lambdal = lambda
          rl = r
          s = 1.0 - rl/ru
          dlambda = (lambdau - lambdal) / s
          lambda = lambdau - dlambda
        } else {
          s = rl/r - 1.0
          if (s < 0.1) s = 0.1
          dlambda = (lambda - lambdal) /s
          lambda_new = 0.75*lambdau + 0.25*lambda
          if (lambda_new > (lambda + dlambda))
            lambda_new = lambda + dlambda
          lambdal = lambda
          rl = r
          lambda = lambda_new
          s = (lambdau - lambdal) / (lambdau - lambda)
        }
      }
      r = ProjectR(x, n, lambda, a, b, c, l, u)
    }
    x
  }

  /** Auxiliary function: solve for absolute value */
  def fabs(x: Double) = if (x < 0) -x else x

  /**
   * Compute A^TA, in order to solve:
   *          argmax{-1/2*lambda x^TA^TAx + x^Tb| x>=0, ||x||_1 = 1}
   * @param a the gradient of R(w)
   * @param b loss - w_dot_grad
   */
  def Update(a: Vector, b: Double) = {
    /** Add a gradient to the gradientSet */
    gradientSet += a
    /** Add a offset to the offsetSet */
    offsetSet += b

    /** update QP hessian matrix and vectors, that is, ATA */
    /** Insert row into Q */
    val rowElem  = mutable.ArrayBuffer[Double]()
    for (elem <- gradientSet.result()){
      rowElem += dot(a, elem)
    }
    Q += rowElem

    /** Insert new column into Q */
    for (i <- 0 until(Q.result().length -1)) {
      Q.result()(i) += dot(gradientSet.result()(i) ,a)
    }

    /** update vectors */
    x += 0
    f += -offsetSet.result()(offsetSet.result().length-1)
  }
}
