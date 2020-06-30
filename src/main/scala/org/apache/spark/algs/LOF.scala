package org.apache.spark.algs

import org.apache.spark.SparkContext
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.rdd.RDD

import scala.collection.mutable.ArrayBuffer

/** Distributed exact implementation of LOF
 *
 * @param data RDD of (index, vector)
 * @param k    Number of neighbors
 * @param numIterations Number of iterations (splits) in case data does not fit in memory (default = 1)
 */
class LOF(data: RDD[(Int, Vector)], k: Int, numIterations: Int) extends kNN_IS(data: RDD[(Int, Vector)], k: Int, numIterations: Int) {

  private def broadcastKNN(knn: Array[(Int, Array[(Int, Float)])], context: SparkContext) = context.broadcast(knn)

  // Calculation of the k nearest neighbors
  val knn_rdd = neighbors()

  /** Calculate the LOF scores in a distributed way
   *
   * @return LOF scores for the entire dataset
   */
  def computeV2() : ArrayBuffer[(Int, Float)] = {
    val knnBroadcast = broadcastKNN(knn_rdd.collect(), data.sparkContext)
    val LOFs = knn_rdd.map(x => x._1).mapPartitions(split => mapLOF2(split, knnBroadcast)).reduce(reduceLOF)
    LOFs.sortBy(_._1)
  }

  /** Calculate the LOF score over the data of each partition.
   *
   * @return LOF scores for the partition
   */
  def mapLOF2(iter: Iterator[Int], knn: Broadcast[Array[(Int, Array[(Int,Float)])]]): Iterator[ArrayBuffer[(Int, Float)]] = {
    var LOFs = new ArrayBuffer[(Int, Float)]

    while(iter.hasNext){
      var i = iter.next()

      val KNN_p = knn.value(i)

      val KNN_oi :Array[(Int,Array[(Int, Float)])] = KNN_p._2.map(oi => {
        (oi._1, knn.value(oi._1)._2)
      })

      val LOF: Float = KNN_oi.map(oi => {
        1 / oi._2.map(qi => {
          Math.max(knn.value(qi._1)._2.maxBy(_._2)._2, qi._2)
        }).sum/k
      }).sum / k / (1 / KNN_p._2.map(oi => {
        Math.max(knn.value(oi._1.toInt)._2.maxBy(_._2)._2, oi._2)
      }).sum / k)

      LOFs += ((i,LOF))
    }


    Iterator.single(LOFs)
  }

  def computeV1(knn_p : Array[(Int, Array[(Int, Float)])] = null) : ArrayBuffer[(Int, Float)] = {

    var knn : Array[(Int, Array[(Int, Float)])] = new Array[(Int, Array[(Int, Float)])](0)
    if (knn_p == null) {
      knn = neighbors().collect()
    }
    else {
      knn = knn_p
    }
    val knnBroadcast = broadcastKNN(knn, data.sparkContext)
    val LOFs = data.mapPartitions(split => mapLOF(split, knnBroadcast)).reduce(reduceLOF)
    LOFs.sortBy(_._1)
  }

  def mapLOF(iter: Iterator[(Int, Vector)], knn: Broadcast[Array[(Int, Array[(Int,Float)])]]): Iterator[ArrayBuffer[(Int, Float)]] = {
    var LOFs = new ArrayBuffer[(Int, Float)]

    while(iter.hasNext){
      var sample = iter.next()

      val KNN_p = knn.value(sample._1)

      val KNN_oi :Array[(Int,Array[(Int, Float)])] = KNN_p._2.map(oi => {
        (oi._1, knn.value(oi._1)._2)
      })

      val LOF: Float = KNN_oi.map(oi => {
        1 / oi._2.map(qi => {
          Math.max(knn.value(qi._1)._2.maxBy(_._2)._2, qi._2)
        }).sum/k
      }).sum / k / (1 / KNN_p._2.map(oi => {
        Math.max(knn.value(oi._1.toInt)._2.maxBy(_._2)._2, oi._2)
      }).sum / k)

      LOFs += ((sample._1,LOF))
    }

//    LOFs.iterator
    Iterator.single(LOFs)
  }

  /**
   * Join the result of the map linking it.
   *
   * @param lof1 A element of the RDD to join
   * @param lof2 Another element of the RDD to join
   * @return Combine of both element
   */
  def reduceLOF(lof1: ArrayBuffer[(Int, Float)], lof2: ArrayBuffer[(Int, Float)]) : ArrayBuffer[(Int,Float)] = {
    lof1++lof2
  }

  /** Calculate the LOF scores locally and sequentially.
   *
   * @return LOF scores for the entire dataset
   */
  def computeOld() : ArrayBuffer[(Int, Float)] = {
    var LOFs = new ArrayBuffer[(Int, Float)]

    val knn = neighbors().collect()

    data.collect().map(sample => {
      val KNN_p = knn(sample._1)

      val KNN_oi :Array[(Int,Array[(Int, Float)])] = KNN_p._2.map(oi => {
        (oi._1, knn(oi._1)._2)
      })

      val LOF: Float = KNN_oi.map(oi => {
        1 / oi._2.map(qi => {
          Math.max(knn(qi._1)._2.maxBy(_._2)._2, qi._2)
        }).sum/k
      }).sum / k / (1 / KNN_p._2.map(oi => {
        Math.max(knn(oi._1.toInt)._2.maxBy(_._2)._2, oi._2)
      }).sum / k)

      LOFs += ((sample._1,LOF))
    })

    LOFs
  }


}
