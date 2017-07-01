package BIDMach

import BIDMat.{Mat,DMat,FMat,FFilter,GMat,GDMat,GFilter}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import org.scalatest._

abstract class BIDMachSpec extends FlatSpec
  with Matchers
  with BeforeAndAfterAll {

  override def beforeAll {
    Mat.checkMKL(false);
  }

  def assert_approx_eq(a: Array[Float], b: Array[Float], eps: Float = 1e-4f) = {
    (a, b).zipped foreach {
      case (x, y) => x should equal (y +- eps)
    }
  }
  
  def assert_approx_eq_double(a: Array[Double], b: Array[Double], eps: Double = 1e-6f) = {
    (a, b).zipped foreach {
      case (x, y) => x should equal (y +- eps)
    }
  }
  
  def checkSimilar(a:DMat, b:DMat, eps:Float):Unit = {
  	a.dims.length should equal (b.dims.length) ;
  	a.dims.data should equal (b.dims.data);
  	assert_approx_eq_double(a.data, b.data, eps);
  }
  
  def checkSimilar(a:DMat, b:DMat):Unit = checkSimilar(a, b, 1e-4f);
  
  def checkSimilar(a:FMat, b:FMat, eps:Float) = {
  	a.dims.length should equal (b.dims.length) ;
  	a.dims.data should equal (b.dims.data);
  	assert_approx_eq(a.data, b.data, eps);
  }
  
  def checkSimilar(a:FMat, b:FMat):Unit = checkSimilar(a, b, 1e-4f);
 
}
