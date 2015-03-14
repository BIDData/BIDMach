package BIDMach.caffe
import BIDMat.{Mat,SBMat,CMat,CSMat,DMat,FMat,FND,GMat,GIMat,GSMat,HMat,Image,IMat,ND,SMat,SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import BIDMach.datasources._
import edu.berkeley.bvlc.SGDSOLVER
import edu.berkeley.bvlc.NET
import edu.berkeley.bvlc.BLOB
import edu.berkeley.bvlc.CAFFE
import scala.collection.immutable.TreeMap
import scala.collection.Iterable

// Caffe Images are W < H < D (< N), Java images are D < W < H, Matlab means file is W < H < D

class Blob(val dims:Array[Int], blob:BLOB) {
	val data = FND(dims);
	val diff = FND(dims);
}

object Blob {
  def apply(blob:BLOB) = {
  	val dims = blob.dims;
  	new Blob(dims, blob);
  }
}



