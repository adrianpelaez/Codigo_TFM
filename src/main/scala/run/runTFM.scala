package org.apache.spark.run

import java.io.{File, PrintWriter}

import jdk.nashorn.internal.ir.Labels
import org.apache.log4j.{Level, Logger}
import org.apache.spark.algs._
import org.apache.spark.mllib.linalg._
import org.apache.spark.sql.{Row, SparkSession}
import org.apache.spark.sql.functions._


object runTFM {

  def evaluate( scores: Array[(Int,Float)], labels: Array[(Int,Float)]) = {
    val scores_sort = scores.sortWith(_._2 > _._2)
    var labels_sort = labels.sortWith(_._2 > _._2)
    var end = false
    var i = 0

    while (end == false) {
      if (labels_sort(i)._2 == 0) {
        end = true
      } else {
        i = i + 1
      }
    }
    val labels_inds = labels_sort.take(i).map(x => x._1).toSet
    val scores_inds = scores_sort.take(i).map(x => x._1).toSet
    val x = labels_inds.intersect(scores_inds)
    x.size
  }

  def main(args: Array[String]): Unit = {

    Logger.getLogger("org").setLevel(Level.ERROR)
    val spark = SparkSession
      .builder
      .appName("runLOF")
      .master("local[*]")
      .getOrCreate()
    val sc = spark.sparkContext

    //val name = "annthyroid"
//    val datasets : Array[String] = Array[String]("annthyroid")
    val datasets : Array[String] = Array[String]("http")
    for (name <- datasets) {
      println(" ====> DATASET:" + name)
      val path = "/home/adrian/dev/TFM/data/csv/" + name + ".csv"
      val df = spark.read.csv(path)
      df.persist
      var data = df.rdd.map { row =>
        Vectors.dense(row.toSeq.toArray.map {
          x => x.toString.toDouble
        })
      }.zipWithIndex().map { x => (x._2.toInt, x._1) }

      val path_labels = "/home/adrian/dev/TFM/data/csv/" + name + "_lab.csv"
      val labels_spark = spark.read.csv(path_labels)
      val labels = labels_spark.rdd.zipWithIndex().map { x => (x._2.toInt, x._1(0).toString.toFloat) }.collect()

      val numIterations = 1
      //val k_values : Array[Int] = Array[Int](2,5,10)//,50,100,200)
      //val k_values : Array[Int] = Array[Int](20)
      val k = 20
      // val maps_values: Array[Int] = Array[Int](1,2,3,4,5,6,7,8,9,10, 20, 50, 100, 150) //,200,250,300)
//       val maps_values: Array[Int] = Array[Int](50,100,200,300,400,500,600,700,800,900,1000,1100,1200,1300,1400)
      val maps_values: Array[Int] = Array[Int](50)
//      data = data.repartition(8)

      // KNN
      /*      val knn = new kNN_IS(data, k, numIterations)
            val resultKNN = knn.kNNScore()
            pw1.write("KNN: " + evaluate(resultKNN, labels))
            val resultKNNAvg = knn.avgkNNScore()
            println("KNN_AVG: " + evaluate(resultKNNAvg, labels))
*/
//            for (maps <- maps_values) {
//              data = data.repartition(maps)
//              val knn = new LocalKNN(data, k)
////              val resultLocalKNN = knn.compute().toArray
////              println("MAPS: " + maps + "LocalKNN: " + evaluate(resultLocalKNN, labels))
//              val resultLocalKNNavg = knn.compute(method = "mean").toArray
//              println("MAPS: " + maps + "LocalKNNavg: " + evaluate(resultLocalKNNavg, labels))
//            }

          //  data = data.repartition(8)

      // LOF
      /*     val lof = new LOF(data, k, numIterations)
           val resultLOF = lof.computeV2().toArray
           println("LOF: " + evaluate(resultLOF, labels))
*/
//           for (maps <- maps_values) {
//             data = data.repartition(maps)
//             val localLOF = new LocalLOF(data, k)
//             val resultLocalLOF = localLOF.compute().toArray
//             println("MAPS: " + maps + " , LocalLOF: " + evaluate(resultLocalLOF, labels))
//           }

      //data = data.repartition(8)

      // LDOF
      /*      val ldof = new LDOF(data, k, numIterations)
            val resultLDOF = ldof.compute()
            println("LDOF: " + evaluate(resultLDOF, labels))
*/
            for (maps <- maps_values) {
              data = data.repartition(maps)
              val localLDOF = new LocalLDOF(data, k)
              val resultLocalLDOF = localLDOF.compute().toArray
              println("MAPS: " + maps + " , LocalLDOF: " + evaluate(resultLocalLDOF, labels))
            }

      //data = data.repartition(8)

      // LoOP
      /*
      val loop = new LoOP(data, k, numIterations, 2)
      val resultLoOP = loop.compute().toArray
      println("LoOP: " + evaluate(resultLoOP, labels))
*/
      for (maps <- maps_values) {
        data = data.repartition(maps)
        val localLoOP = new LocalLoOP(data, k, 2)
        val resultLocalLoOP = localLoOP.compute().toArray
        println("MAPS: " + maps + " , LocalLoOP: " + evaluate(resultLocalLoOP, labels))
      }
/*
      data = data.repartition(8)

      // COF
      val cof = new COF(data, k, numIterations)
      val resultCOF = cof.computeDist().toArray
      println("COF: " + evaluate(resultCOF, labels))
*/
      for (maps <- maps_values) {
        data = data.repartition(maps)
        val localCOF = new LocalCOF(data, k)
        val resultLocalCOF = localCOF.compute()
        println("MAPS: " + maps + " , LocalCOF: " + evaluate(resultLocalCOF, labels))
      }

    }
  }
}
