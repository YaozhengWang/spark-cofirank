package org.apache.spark.mllib.recommendation.cofirank

import org.apache.spark.Logging
import org.apache.spark.ml.recommendation.cofirank.{Cofirank => NewCofirank}
import org.apache.spark.mllib.recommendation.MatrixFactorizationModel
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel

/**
 * Created by cage on 2016/5/23.
 */

/**
* A more compact class to represent a rating
*/
case class Rating(user: Int, product: Int, rating: Double)

class Cofirank private(
   private var numUserBlocks: Int, //user matrix partition
   private var numProductBlocks: Int, //product matrix partition
   private var rank: Int, // features number
   private var iterations: Int,
   private var userLambda: Double,
   private var productLambda: Double,
   private var seed: Long = System.nanoTime()
  ) extends Serializable with Logging{

  /**
   * Constructs an Cofirank instance with default parameters: {numBlocks: -1, rank: 10, iterations:10, lambda:0.01}
   * */
  def this() = this(-1, -1, 10, 10, 0.01, 0.01)

  /** storage level for user/product in/out links*/
  private var intermediateRDDStorageLevel: StorageLevel = StorageLevel.MEMORY_AND_DISK
  private var finalRDDStorageLevel: StorageLevel = StorageLevel.MEMORY_AND_DISK

  /** checkpoint interval */
  private var checkpointInterval: Int = 10

  /**
   *  Set the number of user blocks to parallelize the computation.
   */
  def setUserBlocks(numUserBlocks: Int): this.type ={
    this.numUserBlocks = numUserBlocks
    this
  }

  /**
   * Set the number of product blocks to parallelize the computation
   */
  def setProductBlocks(numProductBlocks: Int): this.type = {
    this.numProductBlocks = numProductBlocks
    this
  }

  /**
   * Set the number of blocks for both user blocks and product blocks to parallelize the computation
   * into: pass -1 for an auto-configured number of blocks. Default: -1
   */
  def setBlocks(numBlocks: Int): this.type = {
    this.numUserBlocks = numBlocks
    this.numProductBlocks = numBlocks
    this
  }

  /**
   * Set the rank of the feature matrices computed(number of features). Default: 10.
   */
  def setRank(rank: Int): this.type = {
    this.rank = rank
    this
  }

  /**
   * Set the number of iterations to run. Default: 10
   */
  def setIterations(iterations: Int): this.type ={
    this.iterations = iterations
    this
  }

  /**
   * Set the regularization parameter of user, userLambda. Default: 0.01
   */
  def setUserLambda(userLambda: Double): this.type ={
    this.userLambda = userLambda
    this
  }

  /**
   * Set the regularization parameter of product, productLambda.Default: 0.01
   */

  def setProductLambda(productLambda: Double): this.type = {
    this.productLambda = productLambda
    this
  }

  /**
   * Set the regularization parameter of user and product, Default:0.01
   */
  def setLambda(lambda: Double): this.type = {
    this.userLambda = lambda
    this.productLambda = lambda
    this
  }

  /**
   * Set a random seed to have deterministic results.
   */
  def setSeed(seed: Long): this.type = {
    this.seed = seed
    this
  }

  /**
   * Set storage level for intermediate RDDs (user/product in/out links). The default value is 'MEMORY_AND_DISK'.
   */
  def setIntermediateRDDStorageLevel(storageLevel: StorageLevel): this.type ={
    require(storageLevel != StorageLevel.NONE, "Cofirank is not designed to run without persisting intermediate RDDs.")
    this.intermediateRDDStorageLevel = storageLevel
    this
  }

  /**
   * Set storage level for final RDDs(user/product used in MatrixFactorizationModel). The default value is 'MEMORY_AND_DISK'.
   */
  def setFinalRDDStorageLevel(storageLevel: StorageLevel): this.type ={
    this.finalRDDStorageLevel = storageLevel
    this
  }

  /**
   * Set period (in iterations) between checkpoints (default = 10). Checkpointing helps with recovery when nodes fail and StackOverflow
   * exceptions caused by long lineage. It also helps with eliminating temporary shuffle files on disk.
   */
  def setCheckpointInterval(checkpointInterval: Int): this.type = {
    this.checkpointInterval = checkpointInterval
    this
  }

  /**
   * Run Cofirank with the configured parameters on an input RDD of (user, product, rating) triples.
   * Returns a MatrixFactorizationModel with feature vectors for each user and product.
   */
  def run(ratings: RDD[Rating]): MatrixFactorizationModel = {
    val sc = ratings.context
    val numUserBlocks = if (this.numUserBlocks == -1) {
      /** numUserBlocks: default value */
      math.max(sc.defaultParallelism, ratings.partitions.size/2)
    } else {
      this.numUserBlocks
    }

    val numProductBlocks = if (this.numProductBlocks == -1){
      /** numProductBlocks: default value */
      math.max(sc.defaultParallelism, ratings.partitions.size / 2)
    } else {
      this.numProductBlocks
    }
    /** The implementation of Cofirank algorithm  which is called */
    val (floatUserFactors, floatProdFactors) = NewCofirank.train[Int](
      ratings = ratings.map(r =>NewCofirank.Rating(r.user, r.product, r.rating.toFloat)),
      rank = rank,
      numUserBlocks = numUserBlocks,
      numItemBlocks = numProductBlocks,
      maxIter = iterations,
      userRegParam = userLambda,
      itemRegParam = productLambda,
      intermediateRDDStorageLevel = intermediateRDDStorageLevel,
      finalRDDStorageLevel = finalRDDStorageLevel,
      checkpointInterval = checkpointInterval,
      seed = seed)

    val userFactors = floatUserFactors.mapValues(_.map(_.toDouble))
      .setName("users")
      .persist(finalRDDStorageLevel)

    val prodFactors = floatProdFactors.mapValues(_.map(_.toDouble))
      .setName("products")
      .persist(finalRDDStorageLevel)

    if (finalRDDStorageLevel != StorageLevel.NONE) {
      userFactors.count()
      prodFactors.count()
    }
    new MatrixFactorizationModel(rank, userFactors, prodFactors)
  }
}

object Cofirank {
  /**
   * Train a matrix factorization model given an RDD of ratings given by users to some products,
   * in the form of (userID, productID, rating) pairs.
   *
   * @param ratings   RDD of (userID, productID, rating) pairs
   * @param rank       number of features to use
   * @param iterations  number of iterations of Cofirank
   * @param userLambda     regularization factor of user(recommended: 0.01)
   * @param productLambda regularization factor of product(recommended: 0.01)
   * @param blocks    level of parallelism to split computation into
   * @param seed      random seed
   */

  def train(
      ratings: RDD[Rating],
      rank: Int,
      iterations: Int,
      userLambda: Double,
      productLambda: Double,
      blocks: Int,
      seed: Long
   ): MatrixFactorizationModel = {
    new Cofirank(blocks, blocks, rank, iterations, userLambda, productLambda, seed).run(ratings)
  }

  /**
   * Train a matrix factorization model given an RDD of ratings given by users to some products,
   * in the form of (userID, productID, rating) pairs.
   *
   * @param  ratings    RDD of (userID, productID, rating) pairs
   * @param rank       number of features to use
   * @param iterations  number of iterations of Cofirank
   * @param userLambda     regularization factor (recommended: 0.01)
   * @param productLambda regularization factor (recommended: 0.01)
   * @param blocks    level of parallelism to split computation into
   */
  def train(
    ratings: RDD[Rating],
    rank: Int,
    iterations: Int,
    userLambda: Double,
    productLambda: Double,
    blocks: Int
  ): MatrixFactorizationModel = {
    new Cofirank(blocks, blocks, rank, iterations, userLambda, productLambda).run(ratings)
  }

  /**
   * Train a matrix factorization model given an RDD of ratings given by users to some products,
   * in the form of (userID, productID, rating) pairs.
   *
   * @param  ratings    RDD of (userID, productID, rating) pairs
   * @param rank       number of features to use
   * @param iterations  number of iterations of Cofirank
   * @param lambda     regularization factor (recommended: 0.01)
   */
  def train(ratings: RDD[Rating], rank: Int, iterations: Int, lambda: Double)
    : MatrixFactorizationModel = {
    train(ratings, rank, iterations, lambda, lambda, -1)
  }

  /**
   * Train a matrix factorization model given an RDD of ratings given by users to some products,
   * in the form of (userID, productID, rating) pairs.
   *
   * @param  ratings    RDD of (userID, productID, rating) pairs
   * @param rank       number of features to use
   * @param iterations  number of iterations of Cofirank
   */
  def train(ratings: RDD[Rating], rank: Int, iterations: Int)
    : MatrixFactorizationModel = {
    train(ratings, rank, iterations, 0.01, 0.01, -1)
  }

  /**
   * Train a matrix factorization model given an RDD of ratings given by users to some products,
   * in the form of (userID, productID, rating) pairs.
   *
   * @param  ratings    RDD of (userID, productID, rating) pairs
   */
  def train(ratings: RDD[Rating]) : MatrixFactorizationModel = {
    train(ratings, 10, 10)
  }
}
