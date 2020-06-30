package org.apache.spark.algs

import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.rdd.RDD
import org.apache.commons.math3.special.Erf
import org.apache.spark.SparkContext
import org.apache.spark.broadcast.Broadcast

import scala.collection.mutable.ArrayBuffer

/** Distributed implementation of LoOP
 *
 * @param data RDD of (index, vector)
 * @param k    Number of neighbors
 * @param numIterations Number of iterations (splits) in case data does not fit in memory (default = 1)
 * @param lambda Parameter for the calculation of the probability
 */
class LoOP(data: RDD[(Int, Vector)], k: Int, numIterations: Int, lambda: Int) extends kNN_IS(data: RDD[(Int, Vector)], k: Int, numIterations: Int) {

  private def broadcastKNN(knn: Array[(Int, Array[(Int, Float)])], context: SparkContext) = context.broadcast(knn)

  // Calculation and broadcast of the k nearest neighbors
  val knn = neighbors().collect()
  var knnBroadcast = broadcastKNN(knn, data.sparkContext)

  /** Calculate the LoOP scores in a distributed way.
   *
   * @return LoOP scores for the entire dataset
   */
  def compute(knn_p : Array[(Int, Array[(Int, Float)])] = null) : ArrayBuffer[(Int, Float)] = {

    // Computes PDIST in a distributed way.
    val pdists = computePdist()
    // Broadcast PDIST. In the calculation of the final score
    // the different nodes need to access this information.
    val pdistBroadcast = data.sparkContext.broadcast(pdists)
    // Computes the LoOP scores in a distributed way.
    val LoOPs = data.mapPartitions(split => mapLoOP(split, knnBroadcast, pdistBroadcast)).reduce(reduceLoOP)
    // Sort result by index.
    LoOPs.sortBy(_._1)
  }

  /** Calculate the PDIST scores in a distributed way.
   *
   * @return Values of PDIST for the entire dataset.
   */
  def computePdist(knn_p : Array[(Int, Array[(Int, Float)])] = null) : ArrayBuffer[Float] = {
    val pdists = data.mapPartitions(split => mapPdist(split, knnBroadcast)).reduce(reducePdist)
    pdists.sortBy(_._1).map(x => x._2)
  }

  /** Calculate PDIST over the data of each partition.
   *
   * @return PDIST for the partition
   */
  def mapPdist(iter: Iterator[(Int, Vector)], knn: Broadcast[Array[(Int, Array[(Int,Float)])]]): Iterator[ArrayBuffer[(Int, Float)]] = {
    var pdist = new ArrayBuffer[(Int, Float)]
    var nplof : Float = 0
    val N = knn.value.length

    //Join the train set
    while (iter.hasNext) {
      val sample = iter.next()
      val neighbors = knn.value(sample._1)
      var ssum = 0.0
      neighbors._2.map(neigh => {
        ssum = ssum + neigh._2 * neigh._2
      })
      pdist.append((sample._1, Math.sqrt(ssum / k).toFloat))
    }
    Iterator.single(pdist)
  }

  /**
   * Join the results of the map phases of the pdist calculation linking it.
   *
   * @param pdist1 A element of the RDD to join
   * @param pdist2 Another element of the RDD to join
   * @return Combine of both element
   */
  def reducePdist(pdist1: ArrayBuffer[(Int, Float)], pdist2: ArrayBuffer[(Int, Float)]) : ArrayBuffer[(Int,Float)] = {
    pdist1++pdist2
  }

  /** Calculate the LoOP score over the data of each partition.
   *
   * @return LoOP scores for the partition
   */
  def mapLoOP(iter: Iterator[(Int, Vector)], knn: Broadcast[Array[(Int, Array[(Int,Float)])]], pdists: Broadcast[ArrayBuffer[Float]]): Iterator[ArrayBuffer[(Int, Float)]] = {

    var aux = new ArrayBuffer[(Int, Vector)]
    var nplof : Float = 0
    val N = knn.value.length

    //Join the train set
    while (iter.hasNext)
      aux.append(iter.next)

    val plofs : ArrayBuffer[Float] = aux.map(sample => {
      val neighbors = knn.value(sample._1)
      var sum = 0.0
      neighbors._2.map(neigh => {
        sum = sum + pdists.value(neigh._1)
      })
      val plof = Math.max((pdists.value(sample._1)*k) / sum, 1)
      nplof = (nplof + (plof-1)*(plof-1)).toFloat
      plof.toFloat
    })
    nplof = lambda*Math.sqrt(nplof/N).toFloat

    val norm = 1 / (nplof*Math.sqrt(2))

    val LoOPs : ArrayBuffer[(Int,Float)] = new ArrayBuffer[(Int,Float)]
    for (i <- 0 until aux.length) {
      LoOPs.append((aux(i)._1,Erf.erf((plofs(i.toInt)-1)*norm).toFloat))
    }

    Iterator.single(LoOPs)
  }

  /**
   * Join the results of the map phases of LoOP calculation linking it.
   *
   * @param loop1 A element of the RDD to join
   * @param loop2 Another element of the RDD to join
   * @return Combine of both element
   */
  def reduceLoOP(loop1: ArrayBuffer[(Int, Float)], loop2: ArrayBuffer[(Int, Float)]) : ArrayBuffer[(Int,Float)] = {
    loop1++loop2
  }

  // Computes the LoOP scores locally and sequentially.
  def computeOld() : Array[(Int, Float)] = {

    val knn = neighbors().collect()
    var nplof : Float = 0

    val pdists : Array[Float] = data.collect().map(sample => {
      val neighbors = knn(sample._1)
      var ssum = 0.0
      neighbors._2.map(neigh => {
        ssum = ssum + neigh._2*neigh._2
      })
      Math.sqrt(ssum/k).toFloat
    })

    val plofs : Array[Float] = data.collect().map(sample => {
      val neighbors = knn(sample._1)
      var sum = 0.0
      neighbors._2.map(neigh => {
        sum = sum + pdists(neigh._1)
      })
      val plof = Math.max((pdists(sample._1)*k) / sum, 1)
      nplof = (nplof + (plof-1)*(plof-1)).toFloat
      plof.toFloat
    })
    nplof = lambda*Math.sqrt(nplof/knn.length).toFloat

    val norm = 1 / (nplof*Math.sqrt(2))
    val LoOPs : Array[(Int,Float)] = data.collect().map(sample => {
      (sample._1,Erf.erf((plofs(sample._1.toInt)-1)*norm).toFloat)
    })
    LoOPs
  }

}
