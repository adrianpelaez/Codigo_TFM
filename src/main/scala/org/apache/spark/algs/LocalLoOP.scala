package org.apache.spark.algs

import org.apache.commons.math3.special.Erf
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.rdd.RDD

import scala.collection.mutable.ArrayBuffer

/** Distributed local implementation of LoOP
 *
 * @param data RDD of (index, vector)
 * @param k    Number of neighbors
 * @param lambda Parameter for the calculation of the probability
 */
class LocalLoOP(data: RDD[(Int, Vector)], k: Int, lambda: Int) extends Serializable {

  def compute(): ArrayBuffer[(Int,Float)] ={
    val LoOPs = data.mapPartitions(split => mapFunction(split)).reduce(reduceFunction)
    LoOPs.sortBy(_._1)
  }


  def mapFunction(iter: Iterator[(Int, Vector)]) : Iterator[ArrayBuffer[(Int, Float)]] = {

    var train = new ArrayBuffer[(Int, Vector)]
    var originalIndex = new ArrayBuffer[Int]

    //Join data set of this partition
    var ind : Int= 0
    while (iter.hasNext) {
      val sample = iter.next
      train.append((ind,sample._2))
      originalIndex.append(sample._1)
      ind+=1
    }

    val size = train.length
    val knnMemb = new KNN(train, k)
    val knn = new Array[Array[(Int, Float)]](size)

    // Calculate KNN for each point in this partition:
    for (i <- 0 until size) {
      knn(i) = knnMemb.neighbors(train(i)._2)
    }
    var nplof : Float = 0

    val pdists : ArrayBuffer[Float] = train.map(sample => {
      val neighbors = knn(sample._1)
      var ssum = 0.0
      neighbors.map(neigh => {
        ssum = ssum + neigh._2*neigh._2
      })
      Math.sqrt(ssum/k).toFloat
    })

    val plofs : ArrayBuffer[Float] = train.map(sample => {
      val neighbors = knn(sample._1)
      var sum = 0.0
      neighbors.map(neigh => {
        sum = sum + pdists(neigh._1)
      })
      val plof = Math.max((pdists(sample._1)*k) / sum, 1)
      nplof = (nplof + (plof-1)*(plof-1)).toFloat
      plof.toFloat
    })
    nplof = lambda*Math.sqrt(nplof/knn.length).toFloat

    val norm = 1 / (nplof*Math.sqrt(2))
    val LoOPs : ArrayBuffer[(Int,Float)] = train.map(sample => {
      (originalIndex(sample._1),Erf.erf((plofs(sample._1.toInt)-1)*norm).toFloat)
    })
    Iterator.single(LoOPs)
  }

  def reduceFunction(mapOut1: ArrayBuffer[(Int, Float)], mapOut2: ArrayBuffer[(Int, Float)]) : ArrayBuffer[(Int, Float)] = {
    mapOut1++mapOut2
  }
}
