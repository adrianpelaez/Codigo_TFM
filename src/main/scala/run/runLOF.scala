package org.apache.spark.run

import org.apache.log4j.{Level, Logger}
import org.apache.spark.algs.LOF
import org.apache.spark.mllib.linalg._
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._


object runLOF {

  def main(args: Array[String]): Unit = {

    Logger.getLogger("org").setLevel(Level.ERROR)
    val spark = SparkSession
      .builder
      .appName("runLOF")
      .master("local[*]")
      .getOrCreate()
    val sc = spark.sparkContext

    // Load dataset
    val path = "/home/adrian/dev/arcelor/data/data_modelling/day=2017-09-09/*"
    val data = spark.read.parquet(path).sort(asc("unix"))
    data.persist

    // Delete String columns
    val dataNumeric = data.drop("unix").drop("BOOMING_JACK_LIMIT_SW").drop("Boomjack_Count").drop("BUCKET_REC_IDENT").drop("TIMESTAMP").drop("MATERIAL_IDENT").drop("LOCATION_SNAME").drop("HAUL_CYCLE_REC_IDENT").drop("maint_id").drop("maint_sub_id")

    //To RDD with indexes
    val dataset = dataNumeric.rdd.map{l =>
      val str = l.toString()
      val cleaned = str.drop(1).dropRight(1)
      Vectors.dense(cleaned.split(",").map(x => x.toDouble))
    }.zipWithIndex().map(f => (f._2.toInt, f._1)).repartition(4).persist()

    // Parameters:
    val k = 5 // number of neighbours
    val numIterations = 1 // number of partitions in the test dataset

    // Compute LOF scores:
    val lof = new LOF(dataset, k, numIterations)
    val resultLOF = lof.computeV2()

  }
}
