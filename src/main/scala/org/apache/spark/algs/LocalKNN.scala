package org.apache.spark.algs

import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.rdd.RDD

import scala.collection.mutable.ArrayBuffer

/** Distributed local implementation of LOF
 *
 * @param data RDD of (index, vector)
 * @param k    Number of neighbors
 */
class LocalKNN(data: RDD[(Int, Vector)], k: Int) extends Serializable {

  def compute(method: String = "largest"): ArrayBuffer[(Int,Float)] ={

    var KNNs = data.mapPartitions(split => mapFunctionLargest(split)).reduce(reduceFunction)

    if (method == "mean") {
      KNNs = data.mapPartitions(split => mapFunctionMean(split)).reduce(reduceFunction)
    }

    KNNs.sortBy(_._1)
  }

  def mapFunctionLargest(iter: Iterator[(Int, Vector)]) : Iterator[ArrayBuffer[(Int, Float)]] = {

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

    var KNNs = new ArrayBuffer[(Int, Float)]
    // Calculate LOF for each point in this partition:
    for (i <- 0 until size) {

      val KNN = knn(i)(k-1)._2

      KNNs.append((originalIndex(i),KNN))
    }
    Iterator.single(KNNs)
  }

  def mapFunctionMean(iter: Iterator[(Int, Vector)]) : Iterator[ArrayBuffer[(Int, Float)]] = {

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

    var KNNs = new ArrayBuffer[(Int, Float)]
    // Calculate LOF for each point in this partition:
    for (i <- 0 until size) {

      val KNN = knn(i).map(y => y._2).sum/k

      KNNs.append((originalIndex(i),KNN))
    }
    Iterator.single(KNNs)
  }

  def reduceFunction(mapOut1: ArrayBuffer[(Int, Float)], mapOut2: ArrayBuffer[(Int, Float)]) : ArrayBuffer[(Int, Float)] = {
    // La forma ideal sería devolver los top N ptos, para no tener que guardar unos resultados del tamaño de la entrada,
    // pero para sacar gráficas con los valores para todos los puntos, por ahora se deja así
    mapOut1++mapOut2
  }
}
