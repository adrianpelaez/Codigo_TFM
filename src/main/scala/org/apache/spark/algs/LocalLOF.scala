package org.apache.spark.algs

import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.rdd.RDD
import scala.collection.mutable.ArrayBuffer

/** Distributed local implementation of LOF
 *
 * @param data RDD of (index, vector)
 * @param k    Number of neighbors
 */
class LocalLOF(data: RDD[(Int, Vector)], k: Int) extends Serializable {

  def compute(): ArrayBuffer[(Int,Float)] ={
    val LOFs = data.mapPartitions(split => mapFunction(split)).reduce(reduceFunction)
    LOFs.sortBy(_._1)
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

    var LOFs = new ArrayBuffer[(Int, Float)]
    // Calculate LOF for each point in this partition:
    for (i <- 0 until size) {
      val KNN_p = knn(i)
      val KNN_oi = KNN_p.map(oi => {
        (oi._1, knn(oi._1))
      })
      val LOF = KNN_oi.map(oi => {
        1 / oi._2.map(qi => {
          Math.max(knn(qi._1).maxBy(_._2)._2, qi._2)
        }).sum/k
      }).sum / k / (1 / KNN_p.map(oi => {
        Math.max(knn(oi._1).maxBy(_._2)._2, oi._2)
      }).sum / k)

      LOFs.append((originalIndex(i),LOF))
    }
    Iterator.single(LOFs)
  }

  def reduceFunction(mapOut1: ArrayBuffer[(Int, Float)], mapOut2: ArrayBuffer[(Int, Float)]) : ArrayBuffer[(Int, Float)] = {
    // La forma ideal sería devolver los top N ptos, para no tener que guardar unos resultados del tamaño de la entrada,
    // pero para sacar gráficas con los valores para todos los puntos, por ahora se deja así
    mapOut1++mapOut2
  }
}
