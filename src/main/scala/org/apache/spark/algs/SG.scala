package org.apache.spark.algs

import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.rdd.RDD

import scala.collection.mutable.ArrayBuffer


class SG(data: RDD[(Int, Vector)]) extends Serializable {

  def compute(): ArrayBuffer[(Int,Float)] ={

    var KNNs = data.mapPartitions(split => mapFunction(split)).reduce(reduceFunction)



    KNNs.sortBy(_._1)
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

    val SCORES = new ArrayBuffer[ArrayBuffer[(Int,Float)]]

    // k values:
    val k_values: Array[Int] = Array[Int](5, 20, 50)


    for ( k <- k_values) {

      val knnMemb = new KNN(train, k)
      val knn = new Array[Array[(Int, Float)]](size)

      // Calculate KNN for each point in this partition:
      for (i <- 0 until size) {
        knn(i) = knnMemb.neighbors(train(i)._2)
      }


      // KNN
      var KNNs = new ArrayBuffer[(Int, Float)]
      for (i <- 0 until size) {
        val KNN = knn(i)(k - 1)._2
        KNNs.append((originalIndex(i), KNN))
      }
      SCORES.append(KNNs)

      // AVG KNN
      var avgKNNs = new ArrayBuffer[(Int, Float)]
      for (i <- 0 until size) {
        val KNN = knn(i).map(y => y._2).sum / k
        avgKNNs.append((originalIndex(i), KNN))
      }
      SCORES.append(avgKNNs)

      // LOF
      var LOFs = new ArrayBuffer[(Int, Float)]
      for (i <- 0 until size) {
        val KNN_p = knn(i)
        val KNN_oi = KNN_p.map(oi => {
          (oi._1, knn(oi._1))
        })
        val LOF = KNN_oi.map(oi => {
          1 / oi._2.map(qi => {
            Math.max(knn(qi._1).maxBy(_._2)._2, qi._2)
          }).sum / k
        }).sum / k / (1 / KNN_p.map(oi => {
          Math.max(knn(oi._1).maxBy(_._2)._2, oi._2)
        }).sum / k)

        LOFs.append((originalIndex(i), LOF))
      }
      SCORES.append(LOFs)

      // LDOF
      var LDOFs = new ArrayBuffer[(Int, Float)]
      for (ind <- 0 until size) {
        val avg_dist = knn(ind).map(y => y._2).sum / k
        var inner_avg_dist =.0.toFloat
        for (i <- 0 until k) {
          for (j <- 0 until k) {
            if (i != j) {
              inner_avg_dist = inner_avg_dist + computeDistance(train(knn(ind)(i)._1)._2, train(knn(ind)(j)._1)._2)
            }
          }
        }
        inner_avg_dist = inner_avg_dist / (k * (k - 1))
        LDOFs.append((originalIndex(ind), avg_dist / inner_avg_dist))
      }
      SCORES.append(LDOFs)

    }
    // Normalizar scores:
    for (i <- 0 until SCORES.size) {
      val norm: Double = SCORES(i).map(x=>x._2).toArray.max
      SCORES(i) = SCORES(i).map(x=>(x._1 , (x._2/norm).toFloat))
    }

    val MEAN = new ArrayBuffer[(Int, Float)]
    for (ind <- 0 until size) {
      val indice = SCORES(0)(ind)._1
      var sum: Float = 0.toFloat
      for (i <- 0 until SCORES.size) {
        sum = sum + SCORES(i)(ind)._2
      }
      MEAN.append((indice, sum/SCORES.size.toFloat))
    }

//    val MAX = new ArrayBuffer[(Int, Float)]
//    for (ind <- 0 until size) {
//      val indice = SCORES(0)(ind)._1
//      var max: Float = 0.toFloat
//      for (i <- 0 until SCORES.size) {
//        if (max < SCORES(i)(ind)._2){
//          max = SCORES(i)(ind)._2
//        }
//      }
//      MAX.append((indice, max))
//    }


    Iterator.single(MEAN)
  }

  def reduceFunction(mapOut1: ArrayBuffer[(Int, Float)], mapOut2: ArrayBuffer[(Int, Float)]) : ArrayBuffer[(Int, Float)] = {
    mapOut1++mapOut2
  }

  def computeDistance(v1: Vector, v2: Vector): Float = {
    math.sqrt(Vectors.sqdist(v1, v2)).toFloat
  }
}
