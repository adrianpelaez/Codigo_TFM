package org.apache.spark.algs

import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.rdd.RDD

import scala.collection.mutable.ArrayBuffer

class LocalCOF(data: RDD[(Int, Vector)], k: Int) extends Serializable {


  def costDescription(knn: (Int, Array[(Int, Float)]), dataArray: ArrayBuffer[(Int,Vector)]): ArrayBuffer[Float] = {
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

  def acDist(knn: (Int, Array[(Int, Float)]), dataArray: ArrayBuffer[(Int,Vector)]): Float = {

    val cost_description = costDescription(knn,dataArray)
    val r = k+1
    var sum: Float = 0

    for (i <- 1 until r) {
      sum = sum + ( (2*(r-i)).toFloat / r.toFloat )*cost_description(i-1)
    }
    sum/(r-1)
  }

  def computeDistance(v1: Vector, v2: Vector): Float = {
    math.sqrt(Vectors.sqdist(v1, v2)).toFloat
  }

  def compute(): Array[(Int,Float)] ={
    val COFs = data.mapPartitions(split => mapFunction(split)).reduce(reduceFunction)
    COFs.sortBy(_._1)
  }

  def mapFunction(iter: Iterator[(Int, Vector)]) : Iterator[Array[(Int, Float)]] = {
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
    val knn = new Array[(Int,Array[(Int, Float)])](size)

    // Calculate KNN for each point in this partition:
    for (i <- 0 until size) {
      knn(i) = (i,knnMemb.neighbors(train(i)._2))
    }

    val COFs : Array[(Int,Float)] = knn.map(n => {
      val ac_d_test = acDist(n,train)
      var sum : Float = n._2.map(x => {
        acDist(knn(x._1),train)
      }).sum
      (originalIndex(n._1), (k * ac_d_test) / sum)
    })
    Iterator.single(COFs)
  }

  def reduceFunction(mapOut1: Array[(Int, Float)], mapOut2: Array[(Int, Float)]) : Array[(Int, Float)] = {
    mapOut1++mapOut2
  }
}
