package org.apache.spark.algs

import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.rdd.RDD

// NOT DISTRIBUTED. This algorithm by nature needs to query data that it cannot access in a distributed environment.

/** Implementation of LDOF
 *
 * @param data RDD of (index, vector)
 * @param k    Number of neighbors
 * @param numIterations Number of iterations (splits) in case data does not fit in memory (default = 1)
 */

class LDOF(data: RDD[(Int, Vector)], k: Int, numIterations: Int) extends kNN_IS(data: RDD[(Int, Vector)], k: Int, numIterations: Int){

  def computeDistance(v1: Vector, v2: Vector): Float = {
    math.sqrt(Vectors.sqdist(v1, v2)).toFloat
  }

  def compute(): Array[(Int,Float)] = {

    val knn = neighbors().collect()
    val dataArray = data.collect()
    val LDOFs : Array[(Int,Float)] = knn.map(x => {
      val avg_dist = x._2.map(y => y._2).sum/k
      var inner_avg_dist = .0.toFloat
      for(i <- 0 until k ) {
        for(j <- 0 until k){
          if(i != j) {
            inner_avg_dist = inner_avg_dist + computeDistance(dataArray(x._2(i)._1)._2,dataArray(x._2(j)._1)._2)
          }
        }
      }
      inner_avg_dist = inner_avg_dist / (k*(k-1))
      (x._1, avg_dist/inner_avg_dist)
    })
    LDOFs
  }
}
