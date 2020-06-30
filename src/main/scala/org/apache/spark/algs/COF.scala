package org.apache.spark.algs

import org.apache.spark.SparkContext
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.rdd.RDD

import scala.collection.mutable.ArrayBuffer

// NOT DISTRIBUTED. This algorithm by nature needs to query data that it cannot access in a distributed environment.

/** Implementation of COF
 *
 * @param data RDD of (index, vector)
 * @param k    Number of neighbors
 * @param numIterations Number of iterations (splits) in case data does not fit in memory (default = 1)
 */
class COF(data: RDD[(Int, Vector)], k: Int, numIterations: Int) extends kNN_IS(data: RDD[(Int, Vector)], k: Int, numIterations: Int){

  private val dataArray = data.sortByKey().collect()
  //private val dataArray = data.collect()

  private def broadcastKNN(knn: Array[(Int, Array[(Int, Float)])], context: SparkContext) = context.broadcast(knn)
  private def broadcastData(dataAux: Array[(Int, Vector)], context: SparkContext) = context.broadcast(dataAux)

  def computeDistance(v1: Vector, v2: Vector): Float = {
    math.sqrt(Vectors.sqdist(v1, v2)).toFloat
  }

  def costDescription(knn: (Int, Array[(Int, Float)])): ArrayBuffer[Float] = {
    var neighbors = knn._2.to[ArrayBuffer]
    var sbn_path: ArrayBuffer[Int] = new ArrayBuffer[Int]()
    var cost_description: ArrayBuffer[Float] = new ArrayBuffer[Float]()
    sbn_path.append(knn._1)

    var next_pto: Int = 0
    var prev_pto: Int = 0
    var next_dist: Float = 0
    var i_remove: Int = 0

    while(!neighbors.isEmpty){
      var min : Float = 999999
      for(x <- sbn_path){
        for(i <- 0 until neighbors.length){
          val dist = computeDistance(dataArray(x)._2, dataArray(neighbors(i)._1)._2)
          if(dist < min){
            min = dist
            prev_pto = x
            next_pto = neighbors(i)._1
            next_dist = dist
            i_remove = i
          }
        }
      }
      sbn_path.append(next_pto)
      cost_description.append(next_dist)
      neighbors.remove(i_remove)
    }
    cost_description
  }

  def acDist(knn: (Int, Array[(Int, Float)])): Float = {

    val cost_description = costDescription(knn)
    val r = k+1
    var sum: Float = 0

    for (i <- 1 until r) {
      sum = sum + ( (2*(r-i)).toFloat / r.toFloat )*cost_description(i-1)
    }
    sum/(r-1)
  }

  def compute(): Array[(Int,Float)] = {
    //val cost_d_test = costDescription(data,test,k)
    val knn = neighbors().collect()
    val COFs : Array[(Int,Float)] = knn.map(n => {
      val ac_d_test = acDist(n)
      var sum : Float = n._2.map(x => {
        acDist(knn(x._1))
      }).sum
      (n._1, (k * ac_d_test) / sum)
    })
    COFs
  }

  def computeDist(): ArrayBuffer[(Int,Float)] = {
    //val cost_d_test = costDescription(data,test,k)
    val knn = neighbors()
    val knnBroadcast = broadcastKNN(knn.collect(), knn.sparkContext)
    val dataBroadcast = broadcastData(dataArray, knn.sparkContext)

    val COFs = knn.mapPartitions(split => mapCOF(split,knnBroadcast,dataBroadcast)).reduce(reduceCOF)

    COFs.sortBy(_._1)
  }

  def mapCOF(iter: Iterator[(Int, Array[(Int,Float)])], knnBroadcast: Broadcast[Array[(Int, Array[(Int,Float)])]], dataBroadcast: Broadcast[Array[(Int, Vector)]]): Iterator[ArrayBuffer[(Int, Float)]] = {

    var COFs = new ArrayBuffer[(Int, Float)]

    while(iter.hasNext) {
      var n = iter.next()
      val ac_d_test = acDistv2(n, dataBroadcast)
      var sum : Float = n._2.map(x => {
        acDistv2(knnBroadcast.value(x._1),dataBroadcast)
      }).sum
      val COF = (n._1, (k * ac_d_test) / sum)
      COFs += COF
    }
    Iterator.single(COFs)
  }

  def reduceCOF(cof1: ArrayBuffer[(Int, Float)], cof2: ArrayBuffer[(Int, Float)]) : ArrayBuffer[(Int,Float)] = {
    cof1++cof2
  }

  def costDescriptionv2(knn: (Int, Array[(Int, Float)]), dataBroadcast: Broadcast[Array[(Int, Vector)]]): ArrayBuffer[Float] = {
    var neighbors = knn._2.to[ArrayBuffer]
    var sbn_path: ArrayBuffer[Int] = new ArrayBuffer[Int]()
    var cost_description: ArrayBuffer[Float] = new ArrayBuffer[Float]()
    sbn_path.append(knn._1)

    var next_pto: Int = 0
    var prev_pto: Int = 0
    var next_dist: Float = 0
    var i_remove: Int = 0

    while(!neighbors.isEmpty){
      var min : Float = 999999
      for(x <- sbn_path){
        for(i <- 0 until neighbors.length){
          val dist = computeDistance(dataBroadcast.value(x)._2, dataBroadcast.value(neighbors(i)._1)._2)
          if(dist < min){
            min = dist
            prev_pto = x
            next_pto = neighbors(i)._1
            next_dist = dist
            i_remove = i
          }
        }
      }
      sbn_path.append(next_pto)
      cost_description.append(next_dist)
      neighbors.remove(i_remove)
    }
    cost_description
  }

  def acDistv2(knn: (Int, Array[(Int, Float)]), dataBroadcast: Broadcast[Array[(Int, Vector)]]): Float = {

    val cost_description = costDescriptionv2(knn,dataBroadcast)
    val r = k+1
    var sum: Float = 0

    for (i <- 1 until r) {
      sum = sum + ( (2*(r-i)).toFloat / r.toFloat )*cost_description(i-1)
    }
    sum/(r-1)
  }
}
