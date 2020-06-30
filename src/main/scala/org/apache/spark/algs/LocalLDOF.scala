package org.apache.spark.algs

import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.rdd.RDD

import scala.collection.mutable.ArrayBuffer


/**  Distributed local implementation of LDOF
 *
 * @param data RDD of (index, vector)
 * @param k    Number of neighbors
 */
class LocalLDOF(data: RDD[(Int, Vector)], k: Int) extends Serializable {

  def compute(): ArrayBuffer[(Int,Float)] ={
    val LDOFs = data.mapPartitions(split => mapFunction(split)).reduce(reduceFunction)
    LDOFs.sortBy(_._1)
  }

  def mapFunction(iter: Iterator[(Int, Vector)]) : Iterator[ArrayBuffer[(Int, Float)]] = {

    var train = new ArrayBuffer[(Int, Vector)]
    var originalIndex = new ArrayBuffer[Int]

    //Join data set of this partition
    var ind: Int = 0
    while (iter.hasNext) {
      val sample = iter.next
      train.append((ind, sample._2))
      originalIndex.append(sample._1)
      ind += 1
    }

    val size = train.length
    val knnMemb = new KNN(train, k)
    val knn = new Array[Array[(Int, Float)]](size)

    // Calculate KNN for each point in this partition:
    for (i <- 0 until size) {
      knn(i) = knnMemb.neighbors(train(i)._2)
    }
    var LDOFs = new ArrayBuffer[(Int, Float)]
    for (ind <- 0 until size) {
        val avg_dist = knn(ind).map(y => y._2).sum/k
        var inner_avg_dist = .0.toFloat
        for(i <- 0 until k ) {
          for(j <- 0 until k){
            if(i != j) {
              inner_avg_dist = inner_avg_dist + computeDistance(train(knn(ind)(i)._1)._2,train(knn(ind)(j)._1)._2)
            }
          }
        }
        inner_avg_dist = inner_avg_dist / (k*(k-1))

        (ind, avg_dist/inner_avg_dist)
        LDOFs.append((originalIndex(ind),avg_dist/inner_avg_dist ))
      }

    Iterator.single(LDOFs)
  }

  def reduceFunction(mapOut1: ArrayBuffer[(Int, Float)], mapOut2: ArrayBuffer[(Int, Float)]) : ArrayBuffer[(Int, Float)] = {
    mapOut1++mapOut2
  }

  def computeDistance(v1: Vector, v2: Vector): Float = {
    math.sqrt(Vectors.sqdist(v1, v2)).toFloat
  }

}
