package org.apache.spark.algs

import org.apache.spark.mllib.linalg.{Vector, Vectors}
import scala.collection.mutable.ArrayBuffer

/** K Nearest Neighbors algorithms
 * The distance used is the Euclidean distance
 *
 * @param train Training set
 * @param k Number of neighbors
 */
class KNN(val train: ArrayBuffer[(Int, Vector)], val k: Int) {

  // Computation of the Euclidean distance between two points
  def computeDistance(v1: Vector, v2: Vector): Float = math.sqrt(Vectors.sqdist(v1, v2)).toFloat

  // Calculation of the k nearest neighbors for a given point
  def neighbors(x: Vector): Array[(Int,Float)] = {
    var nearest = Array.fill(k)(-1)
    var distA = Array.fill(k)(0.0f)
    val size = train.length

    for (i <- 0 until size) {
      var dist : Float = computeDistance(x, train(i)._2)
      if (dist > 0d) {
        var stop = false
        var j = 0
        while (j < k && !stop) {
          if (nearest(j) == (-1) || dist <= distA(j)) {
            for (l <- ((j + 1) until k).reverse) {
              nearest(l) = nearest(l - 1)
              distA(l) = distA(l - 1)
            }
            nearest(j) = i
            distA(j) = dist
            stop = true
          }
          j += 1
        }
      }
    }

    val out: Array[(Int, Float)] = Array.fill(k)(0,.0.toFloat)
    for (i <- 0 until k) {
      out(i) = (train(nearest(i))._1, distA(i))
    }
    out
  }
}