package org.apache.spark.algs

import org.apache.spark.SparkContext
import org.apache.spark.rdd._
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.mllib.linalg.Vector
import scala.collection.mutable.ArrayBuffer

/** Distributed implementation of kNN
 * The distance used is the Euclidean distance
 *
 * @param data RDD of (index, vector)
 * @param k    Number of neighbors
 * @param numIterations Number of iterations (splits) in case data does not fit in memory (default = 1)
 */

class kNN_IS(data: RDD[(Int, Vector)], k: Int, numIterations: Int) extends Serializable {

  private val numSamplesData = data.count()
  private var inc = 0
  private var subdel = 0
  private var topdel = 0
  private var numIter = numIterations

  // Broadcast of the data
  private def broadcastTest(test: Array[(Int, Vector)], context: SparkContext) = context.broadcast(test)

  // Calculation of the k nearest neighbors
  def neighbors(): RDD[(Int, Array[(Int, Float)])] = {

    // Setup:
    subdel = 0
    inc = (numSamplesData / numIter).toInt
    topdel = inc
    if (numIterations == 1) {
      topdel = numSamplesData.toInt + 1
    }

    var testBroadcast: Broadcast[Array[(Int, Vector)]] = null
    var output: RDD[(Int, Array[(Int, Float)])] = null

    // Loop for each iteration of the data (in case the data is splitted)
    for (i <- 0 until numIter) {

      // Broadcast the data (or a chunk of it)
      if (numIter == 1)
        // testBroadcast = broadcastTest(data.sortByKey().collect, data.sparkContext)
      testBroadcast = broadcastTest(data.collect, data.sparkContext)
      else
        testBroadcast = broadcastTest(data.filterByRange(subdel, topdel).collect, data.sparkContext)

      // Process the data partition-wise
      if (output == null) {
        output = data.mapPartitions(split => mapKNN(split, testBroadcast, subdel)).reduceByKey(reduceKNN).cache
      } else {
        output = output.union(data.mapPartitions(split => mapKNN(split, testBroadcast, subdel)).reduceByKey(reduceKNN)).cache
      }
      output.count

      //Update the pairs of delimiters
      subdel = topdel + 1
      topdel = topdel + inc + 1
      testBroadcast.destroy
    }
    output.sortBy(_._1)
  }

  /**
   * Calculate the K nearest neighbor from the test set over the train set.
   *
   * @param iter Iterator of each split of the data set.
   * @param testSet The test set in a broadcast
   * @param subdel Int needed for take order when iterative version is running
   * @return K Nearest Neighbors for this split
   */
  def mapKNN[T](iter: Iterator[(Int, Vector)], testSet: Broadcast[Array[(Int, Vector)]], subdel: Int): Iterator[(Int, Array[(Int, Float)])] = {
    // Initialization
    var train = new ArrayBuffer[(Int, Vector)]
    val size = testSet.value.length

    //Join the train set
    while (iter.hasNext)
      train.append(iter.next)

    val knnMemb = new KNN(train, k)

    var auxSubDel = subdel
    val result = new Array[(Int, Array[(Int, Float)])](size)

    for (i <- 0 until size) {
      // result(i) = (auxSubDel, knnMemb.neighbors(testSet.value(i)._2))
      result(i) = (testSet.value(i)._1, knnMemb.neighbors(testSet.value(i)._2))
      auxSubDel = auxSubDel + 1
    }

    result.iterator
  }

  /**
   * Join the result of the map taking the nearest neighbors.
   *
   * @param mapOut1 A element of the RDD to join
   * @param mapOut2 Another element of the RDD to join
   * @return Combine of both element with the nearest neighbors
   */
  def reduceKNN(mapOut1: Array[(Int, Float)], mapOut2: Array[(Int, Float)]): Array[(Int, Float)] = {

    var itOut1 = 0
    var itOut2 = 0
    var out: Array[(Int, Float)] = new Array[(Int, Float)](k)

    var i = 0
    while (i < k) {
      if (mapOut1(itOut1)._2 <= mapOut2(itOut2)._2) {
        out(i) = (mapOut1(itOut1)._1, mapOut1(itOut1)._2)
        if (mapOut1(itOut1)._2 == mapOut2(itOut2)._2) {
          i += 1
          if (i < k) {
            out(i) = (mapOut2(itOut2)._1, mapOut2(itOut2)._2)
            itOut2 = itOut2 + 1
          }
        }
        itOut1 = itOut1 + 1

      } else {
        out(i) = (mapOut2(itOut2)._1, mapOut2(itOut2)._2)
        itOut2 = itOut2 + 1
      }
      i += 1
    }
    out
  }

  /**
   * Obtain the anomaly score as the distance to the nearest neighboring k in a distributed way.
   *
   * @return Array with the scores.
   */
  def kNNScore() : Array[(Int,Float)] = {
    val knn = neighbors()
    val output = knn.map(x => {
      (x._1,x._2(k-1)._2)
    })
    output.collect()
  }

  /**
   * Obtain the anomaly score as the average distance to to its k-neighborhood in a distributed way.
   *
   * @return Array with the scores.
   */
  def avgkNNScore() : Array[(Int,Float)] = {
    val knn = neighbors()
    val output = knn.map(x => {

      (x._1,x._2.map(y => y._2).sum/k)
    })
    output.collect()
  }
}
