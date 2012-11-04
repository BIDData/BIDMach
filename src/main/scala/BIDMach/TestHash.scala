package BIDMach
import BIDMat.{Mat,BMat,CMat,CSMat,DMat,FMat,IMat,HMat,GMat,GIMat,GSMat,SMat,SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import Learner._

object TestHash {
  def main(args: Array[String]): Unit = {

    val dirname = "d:\\sentiment\\sorted_data\\books\\"
  	val smap:CSMat = load(dirname+"sparsemat.mat", "smap")
  	println("done "+size(smap,1)+" "+size(smap,2))
  }
}
