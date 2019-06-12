package BIDMach

import org.scalatest._

abstract class BIDMachSpec extends FlatSpec
  with Matchers
  with BeforeAndAfterAll {

  override def beforeAll {
    BIDMat.Mat.checkMKL(false);
  }

  def assert_approx_eq(a: Array[Float], b: Array[Float], eps: Float = 1e-4f) = {
    (a, b).zipped foreach {
      case (x, y) => { 
        val scale = (math.abs(x) + math.abs(y) + eps).toFloat;
        x / scale should equal ((y / scale) +- eps)
      }
    }
  }
  
  def assert_approx_eq_double(a: Array[Double], b: Array[Double], eps: Double = 1e-6f) = {
    (a, b).zipped foreach {
      case (x, y) => { 
        val scale = (math.abs(x) + math.abs(y) + eps);
        x / scale should equal ((y / scale) +- eps)
      }
    }
  }
  
}
