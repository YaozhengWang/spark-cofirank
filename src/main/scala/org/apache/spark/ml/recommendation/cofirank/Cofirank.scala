package org.apache.spark.ml.recommendation.cofirank

import org.apache.spark.annotation.DeveloperApi
import org.apache.spark.mllib.optimization.bmrm.{DaiFletcherUpdater, NdcgSubGradient, BMRM}
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel
import org.apache.spark.Partitioner
import org.apache.spark.util.collection.{OpenHashSet, OpenHashMap}
import org.apache.spark.util.random.XORShiftRandom
import scala.reflect.ClassTag
import scala.collection.mutable
import scala.util.Sorting
import scala.util.hashing.byteswap64
import org.apache.spark.util.collection.{Sorter, SortDataFormat}
import com.github.fommil.netlib.BLAS.{getInstance => blas}
import org.apache.hadoop.fs.{FileSystem, Path}
import java.io.IOException
import org.apache.spark.mllib.linalg.{Vectors, Vector}

/**
 * Created by cage on 2016/5/23.
 */

class Cofirank {

}

@DeveloperApi
object Cofirank {
  /**
   *  Rating class for better code readability
   */
  @DeveloperApi
  case class Rating[@specialized(Int, Long) ID](user: ID, item: ID, rating: Float)

  @DeveloperApi
  def train[ID: ClassTag](
      ratings: RDD[Rating[ID]],
      rank: Int = 10,
      numUserBlocks: Int = 10,
      numItemBlocks: Int = 10,
      maxIter: Int = 10,
      userRegParam: Double = 1.0,
      itemRegParam: Double = 1.0,
      intermediateRDDStorageLevel: StorageLevel = StorageLevel.MEMORY_AND_DISK,
      finalRDDStorageLevel: StorageLevel = StorageLevel.MEMORY_AND_DISK,
      checkpointInterval: Int = 10,
      seed: Long = 0L) (implicit ord: Ordering[ID]): (RDD[(ID, Array[Float])], RDD[(ID, Array[Float])]) = {
    require(intermediateRDDStorageLevel != StorageLevel.NONE, "Cofirank is not designed to run without persisting intermediate RDDs.")
    val sc = ratings.sparkContext
    val userPart = new CofirankPartitioner(numUserBlocks)
    val itemPart = new CofirankPartitioner(numItemBlocks)
    val userLocalIndexEncoder = new LocalIndexEncoder(userPart.numPartitions)
    val itemLocalIndexEncoder = new LocalIndexEncoder(itemPart.numPartitions)
    /** Divide ratings into userPart.numPartitions * itemPart.numPartitions blocks, blockRatings: ((userBlockId, itemBlockId), ratingBlock) */
    val blockRatings = partitionRatings(ratings, userPart, itemPart).persist(intermediateRDDStorageLevel)
    val (userInBlocks, userOutBlocks) = makeBlocks("user", blockRatings, userPart, itemPart, intermediateRDDStorageLevel)
    userOutBlocks.count()
    val swappedBlockRatings = blockRatings.map{
      case ((userBlockId, itemBlockId), RatingBlock(userIds, itemIds, localRatings)) =>
        ((itemBlockId, userBlockId), RatingBlock(itemIds, userIds, localRatings))
    }
    val (itemInBlocks, itemOutBlocks) = makeBlocks("item", swappedBlockRatings, itemPart, userPart, intermediateRDDStorageLevel)
    itemOutBlocks.count()
    val seedGen = new XORShiftRandom(seed)
    /** Initialize user matrix */
    var userFactors = initialize(userInBlocks, rank, seedGen.nextLong())
    /** Initialize item matrix */
    var itemFactors = initialize(itemInBlocks, rank, seedGen.nextLong())
    var previousCheckpointFile: Option[String] = None
    val shouldCheckpoint: Int => Boolean = (iter) =>
      sc.checkpointDir.isDefined && checkpointInterval != -1 && (iter % checkpointInterval == 0)
    val deletePreviousCheckpointFile: () => Unit = () =>
      previousCheckpointFile.foreach { file =>
        try {
          FileSystem.get(sc.hadoopConfiguration).delete(new Path(file), true)
        } catch {
          case e: IOException =>
            println(s"Cannot delete checkpoint file $file:", e)
        }
      }
    /** Algorithm iterative process */
    for (iter <-0 until maxIter) {
      userFactors = computeFactors(itemFactors, itemOutBlocks, userInBlocks, userFactors, rank, userRegParam, itemLocalIndexEncoder)
      if (shouldCheckpoint(iter)) {
        userFactors.checkpoint()
        /** checkpoint item factors and cut lineage */
        userFactors.count()
        deletePreviousCheckpointFile()
        previousCheckpointFile = userFactors.getCheckpointFile
      }
      itemFactors = computeFactors(userFactors, userOutBlocks, itemInBlocks, itemFactors, rank, itemRegParam, userLocalIndexEncoder)
    }

    val userIdAndFactors = userInBlocks.mapValues(_.srcIds).join(userFactors)
      .mapPartitions({ items =>
        items.flatMap { case (_, (ids, factors)) =>
          ids.view.zip(factors)
        }
      }, preservesPartitioning = true).setName("userFactors").persist(finalRDDStorageLevel)

    val itemIdAndFactors = itemInBlocks.mapValues(_.srcIds).join(itemFactors)
      .mapPartitions({ items =>
      items.flatMap {case (_, (ids, factors)) =>
        ids.view.zip(factors)
      }
    }, preservesPartitioning = true).setName("itemFactors").persist(finalRDDStorageLevel)

    if (finalRDDStorageLevel != StorageLevel.NONE) {
      userIdAndFactors.count()
      itemFactors.unpersist()
      itemIdAndFactors.count()
      userInBlocks.unpersist()
      userOutBlocks.unpersist()
      itemInBlocks.unpersist()
      itemOutBlocks.unpersist()
      blockRatings.unpersist()
    }
    (userIdAndFactors, itemIdAndFactors)
  }

  private def computeFactors[ID](
    srcFactorBlocks: RDD[(Int, FactorBlock)],
    srcOutBlocks: RDD[(Int, OutBlock)],
    dstInBlocks: RDD[(Int, InBlock[ID])],
    dstFactorBlocks: RDD[(Int, FactorBlock)],
    rank: Int,
    regParam:Double,
    srcEncoder: LocalIndexEncoder): RDD[(Int, FactorBlock)] = {
    val numSrcBlocks = srcFactorBlocks.partitions.length
    val srcOut = srcOutBlocks.join(srcFactorBlocks).flatMap{
      case (srcBlockId, (srcOutBlock, srcFactors)) =>
        srcOutBlock.view.zipWithIndex.map {case (activeIndices, dstBlockId) =>
          (dstBlockId, (srcBlockId, activeIndices.map(idx => srcFactors(idx))))
        }
    }
    val merged = srcOut.groupByKey(new CofirankPartitioner(dstInBlocks.partitions.length))
    val dstOut = dstInBlocks.join(dstFactorBlocks)
    dstOut.join(merged).mapValues {
      case ((InBlock(dstIds, srcPtrs, srcEncodedIndices, ratings), dst_factors), srcFactors) =>
        val sortedSrcFactors = new Array[FactorBlock](numSrcBlocks)
        srcFactors.foreach{case (srcBlockId, factors) => sortedSrcFactors(srcBlockId) = factors}
        val dstFactors = new Array[Array[Float]](dstIds.length)
        var j = 0
        while (j < dstIds.length){
          var i = srcPtrs(j)
          val data = mutable.ArrayBuilder.make[Vector]
          val label = mutable.ArrayBuilder.make[Double]
          while (i < srcPtrs(j + 1)){
            val encoded = srcEncodedIndices(i)
            val blockId = srcEncoder.blockId(encoded)
            val localIndex = srcEncoder.localIndex(encoded)
            val srcFactor = sortedSrcFactors(blockId)(localIndex)
            val rating = ratings(i)
            //val vFactor = Array.fill[Double](srcFactor.length)(0.0)
            //(0 until srcFactor.length).map{i => vFactor(i) = srcFactor(i).toDouble}
            data += Vectors.dense(srcFactor.map(_.toDouble))
            label += rating
            i += 1
          }
          val bmrm = new BMRM(new NdcgSubGradient(), new DaiFletcherUpdater())
         // val dstfact = dst_factors(j)
         // val vdstfact = Array.fill[Double](dstfact.length)(0.0)
         // (0 until vdstfact.length).map{i => vdstfact(i) = dstfact(i).toDouble}
          dstFactors(j) = bmrm.optimize(data.result(), label.result(),
            Vectors.dense(dst_factors(j).map(_.toDouble))).toArray.map(_.toFloat)
          //val dstF_final = Array.fill[Float](dstF.length)(0)
          //(0 until dstF.length).map(i => dstF_final(i) = dstF(i).toFloat)
          //dstFactors(j) = dstF_final
          j += 1
        }
        dstFactors
    }
  }

  /**
   * Factor block that stores factors (Array[Float]) in an Array.
   */
  private type FactorBlock = Array[Array[Float]]
  /**
   * Initializes factors randomly given the in-link blocks.
   *
   * @param inBlocks in-link blocks
   * @param rank rank
   * @return initialized factor blocks
   */

  private def initialize[ID](inBlocks: RDD[(Int, InBlock[ID])], rank: Int, seed: Long): RDD[(Int, FactorBlock)] = {
    inBlocks.map {case (srcBlockId, inBlock) =>
        val random = new XORShiftRandom(byteswap64(seed ^ srcBlockId))
        val factors = Array.fill(inBlock.srcIds.length) {
          val factor = Array.fill(rank)(random.nextGaussian().toFloat)
          val nrm = blas.snrm2(rank, factor, 1)
          blas.sscal(rank, 1.0f / nrm, factor, 1)
          factor
        }
      (srcBlockId, factors)
    }
  }

  /**
   * In-link block for computing src (user/item) factors. This includes the original src IDs
   * of the elements within this block as well as encoded dst (item/user) indices and corresponding
   * ratings. The dst indices are in the form of (blockId, localIndex), which are not the original
   * dst IDs. To compute src factors, we expect receiving dst factors that match the dst indices.
   * For example, if we have an in-link record
   *
   * {srcId: 0, dstBlockId: 2, dstLocalIndex: 3, rating: 5.0},
   *
   * and assume that the dst factors are stored as dstFactors: Map[Int, Array[Array[Float]]], which
   * is a blockId to dst factors map, the corresponding dst factor of the record is dstFactor(2)(3).
   *
   * We use a CSC-like (compressed sparse column) format to store the in-link information. So we can
   * compute src factors one after another using only one normal equation instance.
   *
   * @param srcIds src ids (ordered)
   * @param dstPtrs dst pointers. Elements in range [dstPtrs(i), dstPtrs(i+1)) of dst indices and
   *                ratings are associated with srcIds(i).
   * @param dstEncodedIndices encoded dst indices
   * @param ratings ratings
   *
   * @see [[LocalIndexEncoder]]
   */
  private case class InBlock[@specialized(Int, Long) ID: ClassTag](
     srcIds: Array[ID],
     dstPtrs: Array[Int],
     dstEncodedIndices: Array[Int],
     ratings: Array[Float]) {
    /** Size of the block. */
    def size: Int = ratings.length
    require(dstEncodedIndices.length == size)
    require(dstPtrs.length == srcIds.length + 1)
  }
  /**
   * Out-link block that stores, for each dst (item/user) block, which src (user/item) factors to
   * send. For example, outLinkBlock(0) contains the local indices (not the original src IDs) of the
   * src factors in this block to send to dst block 0.
   */
  private type OutBlock = Array[Array[Int]]

  /**
   * Builder for uncompressed in-blocks of (srcId, dstEncodeIndex, rating) tuples.
   * @param encoder encoder for dst indices
   */
  private class UncompressedInBlockBuilder[@specialized(Int, Long)ID: ClassTag](encoder:
       LocalIndexEncoder)(implicit ord: Ordering[ID]) {
    private val srcIds = mutable.ArrayBuilder.make[ID]
    private val dstEncodedIndices = mutable.ArrayBuilder.make[Int]
    private val ratings = mutable.ArrayBuilder.make[Float]

    /**
     * Adds a dst block of (srcId, dstLocalIndex, rating) tuples.
     *
     * @param dstBlockId dst block Id
     * @param srcIds original src IDs
     * @param dstLocalIndices dst local indices
     * @param ratings ratings
     */
    def add(dstBlockId: Int, srcIds: Array[ID],
             dstLocalIndices: Array[Int], ratings: Array[Float]): this.type = {
      val sz = srcIds.length
      require(dstLocalIndices.length == sz)
      require(ratings.length == sz)
      this.srcIds ++= srcIds
      this.ratings ++= ratings
      var j = 0
      while (j < sz) {
        this.dstEncodedIndices += encoder.encode(dstBlockId, dstLocalIndices(j))
        j += 1
      }
      this
    }

    /** Builds a [[UncompressedInBlock]]. */
    def build():UncompressedInBlock[ID] = {
      new UncompressedInBlock(srcIds.result(), dstEncodedIndices.result(), ratings.result())
    }
  }

  /**
   * A block of (srcId, dstEncodeIndex, rating) tuples stored in primitive arrays.
   */
  private class UncompressedInBlock[@specialized(Int, Long)ID: ClassTag](
      val srcIds: Array[ID],
      val dstEncodedIndices: Array[Int],
      val ratings: Array[Float])(implicit ord: Ordering[ID]) {
    /** Size the of block. */
    def length: Int = srcIds.length

    /**
     * Compresses the block into an [[InBlock]].
     * Sorting is done using Spark's built-in Timsort to avoid generating too many objects.
     */
    def compress(): InBlock[ID] = {
      val sz = length
      assert(sz > 0, "Empty in-link block should not exist.")
      /** crucial step: sort [[UncompressedInBlock]],sortedkey = srcIds*/
      sort()
      val uniqueSrcIdsBuilder = mutable.ArrayBuilder.make[ID]
      val dstCountsBuilder = mutable.ArrayBuilder.make[Int]
      var preSrcId = srcIds(0)
      uniqueSrcIdsBuilder += preSrcId
      var curCount = 1
      var i = 1
      var j = 0
      while (i < sz) {
        val srcId = srcIds(i)
        if (srcId != preSrcId){
          uniqueSrcIdsBuilder += srcId
          dstCountsBuilder += curCount
          preSrcId = srcId
          j += 1
          curCount = 0
        }
        curCount += 1
        i += 1
      }
      dstCountsBuilder += curCount
      val uniqueSrcIds = uniqueSrcIdsBuilder.result()
      val numUniqueSrcIds = uniqueSrcIds.length
      val dstCounts = dstCountsBuilder.result()
      val dstPtrs = new Array[Int](numUniqueSrcIds + 1)
      var sum = 0
      i = 0
      while (i < numUniqueSrcIds) {
        sum += dstCounts(i)
        i += 1
        dstPtrs(i) = sum
      }
      InBlock(uniqueSrcIds, dstPtrs, dstEncodedIndices, ratings)
    }

    private def sort(): Unit ={
      val sorter = new Sorter(new UncompressedInBlockSort[ID])
      sorter.sort(this, 0, length, Ordering[KeyWrapper[ID]])
    }
  }

  /**
   * A wrapper that holds a primitive key.
   *
   * @see [[UncompressedInBlockSort]]
   */
  private class KeyWrapper[@specialized(Int, Long) ID: ClassTag](
      implicit ord: Ordering[ID]) extends Ordered[KeyWrapper[ID]] {

    var key: ID = _

    override def compare(that: KeyWrapper[ID]): Int = {
      ord.compare(key, that.key)
    }

    def setKey(key: ID): this.type = {
      this.key = key
      this
    }
  }

  /**
   * [[SortDataFormat]] of [[UncompressedInBlock]] used by [[Sorter]].
   */
  private class UncompressedInBlockSort[@specialized(Int, Long) ID: ClassTag](
      implicit ord: Ordering[ID])
    extends SortDataFormat[KeyWrapper[ID], UncompressedInBlock[ID]] {

    override def newKey(): KeyWrapper[ID] = new KeyWrapper()

    override def getKey(
        data: UncompressedInBlock[ID],
        pos: Int,
        reuse: KeyWrapper[ID]): KeyWrapper[ID] = {
      if (reuse == null) {
        new KeyWrapper().setKey(data.srcIds(pos))
      } else {
        reuse.setKey(data.srcIds(pos))
      }
    }

    override def getKey(
        data: UncompressedInBlock[ID],
        pos: Int): KeyWrapper[ID] = {
      getKey(data, pos, null)
    }

    private def swapElements[@specialized(Int, Float) T](
        data: Array[T],
        pos0: Int,
        pos1: Int): Unit = {
      val tmp = data(pos0)
      data(pos0) = data(pos1)
      data(pos1) = tmp
    }

    override def swap(data: UncompressedInBlock[ID], pos0: Int, pos1: Int): Unit = {
      swapElements(data.srcIds, pos0, pos1)
      swapElements(data.dstEncodedIndices, pos0, pos1)
      swapElements(data.ratings, pos0, pos1)
    }

    override def copyRange(
        src: UncompressedInBlock[ID],
        srcPos: Int,
        dst: UncompressedInBlock[ID],
        dstPos: Int,
        length: Int): Unit = {
      System.arraycopy(src.srcIds, srcPos, dst.srcIds, dstPos, length)
      System.arraycopy(src.dstEncodedIndices, srcPos, dst.dstEncodedIndices, dstPos, length)
      System.arraycopy(src.ratings, srcPos, dst.ratings, dstPos, length)
    }

    override def allocate(length: Int): UncompressedInBlock[ID] = {
      new UncompressedInBlock(
        new Array[ID](length), new Array[Int](length), new Array[Float](length))
    }

    override def copyElement(
        src: UncompressedInBlock[ID],
        srcPos: Int,
        dst: UncompressedInBlock[ID],
        dstPos: Int): Unit = {
      dst.srcIds(dstPos) = src.srcIds(srcPos)
      dst.dstEncodedIndices(dstPos) = src.dstEncodedIndices(srcPos)
      dst.ratings(dstPos) = src.ratings(srcPos)
    }
  }

  /**
   * Creates in-blocks and out-blocks from rating blocks.
   * @param prefix prefix for in/out-block names
   * @param ratingBlocks rating blocks
   * @param srcPart partitioner for src IDs
   * @param dstPart partitioner for dst IDs
   * @param storageLevel intermediateRDDStorageLevel
   * @return (InBlock, OutBlock)
   */
  private def makeBlocks[ID: ClassTag](
     prefix: String,
     ratingBlocks: RDD[((Int, Int), RatingBlock[ID])],
     srcPart: Partitioner,
     dstPart: Partitioner,
     storageLevel: StorageLevel)(implicit srcOrd: Ordering[ID]): (RDD[(Int, InBlock[ID])], RDD[(Int, OutBlock)]) = {
    val inBlocks = ratingBlocks.map {
      case  ((srcBlockId, dstBlockId), RatingBlock(srcIds, dstIds, ratings)) =>
        val dstIdSet = new OpenHashSet[ID]( 1 << 20)
        /** Add the elements of dstIds to the dstIdSet and remove duplicate elements.*/
        dstIds.foreach(dstIdSet.add)
        val sortedDstIds = new Array[ID](dstIdSet.size)
        var i = 0
        var pos = dstIdSet.nextPos(0)
        while (pos != -1) {
          sortedDstIds(i) = dstIdSet.getValue(pos)
          pos = dstIdSet.nextPos(pos + 1)
          i += 1
        }

        assert(i == dstIdSet.size)
        /** Sort the array sortedDstIds*/
        Sorting.quickSort(sortedDstIds)
        val dstIdToLocalIndex = new OpenHashMap[ID, Int](sortedDstIds.length)
        i = 0
        while (i < sortedDstIds.length) {
          dstIdToLocalIndex.update(sortedDstIds(i), i)
          i += 1
        }
        val dstLocalIndices = dstIds.map(dstIdToLocalIndex.apply)
        (srcBlockId, (dstBlockId, srcIds, dstLocalIndices, ratings))
    }.groupByKey(new CofirankPartitioner(srcPart.numPartitions))
     .mapValues { iter =>
       val builder = new UncompressedInBlockBuilder[ID](new LocalIndexEncoder(dstPart.numPartitions))
       iter.foreach{case (dstBlockId, srcIds, dstLocalIndices, ratings) =>
           builder.add(dstBlockId, srcIds, dstLocalIndices, ratings)
       }
       builder.build().compress()
     }.setName(prefix + "InBlocks").persist(storageLevel)
    val outBlocks = inBlocks.mapValues{ case InBlock(srcIds, dstPtrs, dstEncodeIndices, _) =>
        val encoder = new LocalIndexEncoder(dstPart.numPartitions)
        val activeIds = Array.fill(dstPart.numPartitions)(mutable.ArrayBuilder.make[Int])
        var i = 0
        val seen = new Array[Boolean](dstPart.numPartitions)
        while (i < srcIds.length) {
          var j = dstPtrs(i)
          java.util.Arrays.fill(seen, false)
          while (j < dstPtrs(i + 1)) {
            val dstBlockId = encoder.blockId(dstEncodeIndices(j))
            if (!seen(dstBlockId)) {
              activeIds(dstBlockId) += i
              seen(dstBlockId) = true
            }
            j += 1
          }
          i += 1
        }
        activeIds.map{ x =>
          x.result()
        }
    }.setName(prefix + "OutBlocks").persist(storageLevel)
    (inBlocks, outBlocks)
  }

  /**
   * A rating block that contains src IDs, dst IDs, and ratings, stored in primitive arrays.
   */
  private case class RatingBlock[@specialized(Int, Long) ID: ClassTag](
     srcIds: Array[ID],
     dstIds: Array[ID],
     ratings: Array[Float]) {
    /** Size of the block. */
    def size: Int = srcIds.length
    require(dstIds.length == srcIds.length)
    require(ratings.length == srcIds.length)
  }

  /**
   * Builder for [[RatingBlock]]. [[mutable.ArrayBuilder]] is used to avoid boxing/unboxing.
   */
  private class RatingBlockBuilder[@specialized(Int, Long) ID: ClassTag] extends Serializable {
    /** Define srcIds, dstIds, ratings. */
    private val srcIds = mutable.ArrayBuilder.make[ID]
    private val dstIds = mutable.ArrayBuilder.make[ID]
    private val ratings = mutable.ArrayBuilder.make[Float]
    var size = 0

    /** Add a rating. */
    def add(r: Rating[ID]): this.type = {
      size += 1
      srcIds += r.user
      dstIds += r.item
      ratings += r.rating
      this
    }

    /** Merges another [[RatingBlockBuilder]].*/
    def merge(other: RatingBlock[ID]): this.type = {
      size += other.srcIds.length
      srcIds ++= other.srcIds
      dstIds ++= other.dstIds
      ratings ++= other.ratings
      this
    }

    /** Builds a [[RatingBlock]]. */
    def build(): RatingBlock[ID] = {
      RatingBlock[ID](srcIds.result(), dstIds.result(), ratings.result())
    }
  }

  /**
   * Partitions raw ratings into blocks.
   *
   * @param ratings raw ratings
   * @param srcPart partitioner for src IDs
   * @param dstPart partitioner for dst IDs
   *
   * @return an RDD of rating blocks in the form of ((srcBlockId, dstBlockId), ratingBlock)
   */

  private def partitionRatings[ID: ClassTag](
      ratings: RDD[Rating[ID]],
      srcPart: Partitioner,
      dstPart: Partitioner
  ): RDD[((Int, Int), RatingBlock[ID])] = {
    /** The number of ratings' partitions */
    val numPartitions = srcPart.numPartitions * dstPart.numPartitions
    ratings.mapPartitions{ iter =>
      val builders = Array.fill(numPartitions)(new RatingBlockBuilder[ID])
      iter.flatMap{ r =>
        val srcBlockId = srcPart.getPartition(r.user)
        val dstBlockId = dstPart.getPartition(r.item)
        val idx = srcBlockId + srcPart.numPartitions * dstBlockId
        val builder = builders(idx)
        builder.add(r)
        if (builder.size >= 2048) {
          builders(idx) = new RatingBlockBuilder
          Iterator.single(((srcBlockId, dstBlockId), builder.build()))
        } else {
          Iterator.empty
        }
      } ++ {
        builders.view.zipWithIndex.filter(_._1.size > 0).map { case (block, idx) =>
          val srcBlockId = idx % srcPart.numPartitions
          val dstBlockId = idx / srcPart.numPartitions
          ((srcBlockId, dstBlockId), block.build())
        }
      }
    }.groupByKey().mapValues{blocks =>
      val builder = new RatingBlockBuilder[ID]
      blocks.foreach(builder.merge)
      builder.build()
    }.setName("ratingBlocks")
  }

  /**
   *  Encoder for storing (blockId, localIndex) into a single integer.
   *
   *  We use the leading bits (including the sign bit) to store the block id and the rest to store
   *  the local index. This is based on the assumption that users/items are approximately evenly partitioned.
   *  With this assumption, we should be able to encode two billion distinct values.
   *
   *  @param numBlocks number of blocks
   */
  private class LocalIndexEncoder(numBlocks: Int) extends Serializable {
    require(numBlocks > 0, s"numBlocks must be positive but found $numBlocks.")
    /** compute the number of bits for localIndex */
    private[this] final val numLocalIndexBits =
      math.min(java.lang.Integer.numberOfLeadingZeros(numBlocks - 1), 31)
    private[this] final val localIndexMask = (1 << numLocalIndexBits) - 1

    /** Encodes a (blockId, localIndex) into a single integer. */
    def encode(blockId: Int, localIndex: Int): Int = {
        require(blockId < numBlocks)
        require((localIndex & ~localIndexMask) == 0)
        (blockId << numLocalIndexBits) | localIndex
    }

    /** Get the block id from an encoded index. */
    def blockId(encoded: Int): Int = {
      encoded >>> numLocalIndexBits
    }

    /** Get the local index from an encoded index. */
    def localIndex(encoded: Int): Int = {
      encoded & localIndexMask
    }
  }

  /**
   * Partitioner used by Cofirank. We require that getPartition is a projection. That is, for any key k,
   * we have getPartition(getPartition(k)) = getPartition(k). Since the default HashPartitioner satisfies
   * this requirement, we simple use a type alias here.
   */
  private type CofirankPartitioner = org.apache.spark.HashPartitioner
}
