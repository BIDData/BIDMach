package BIDMach.datasources
import BIDMat.{Mat,SBMat,CMat,CSMat,DMat,FMat,IMat,HMat,GMat,GIMat,GSMat,SMat,SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import java.io._

class StackedDS(val s1:DataSource, val s2:DataSource, 
    override val opts:DataSource.Opts = new DataSource.Options) extends DataSource(opts) {

  omats = null
  
  def init = {
    s1.init
    s2.init
    val mats1 = s1.omats
    val mats2 = s2.omats
    omats = new Array[Mat](mats1.length + mats2.length)
    for (i <- 0 until mats1.length) {
      omats(i) = mats1(i);
    }
    for (i <- 0 until mats2.length) {
      omats(i+mats1.length) = mats2(i);
    } 
  }
  
  def nmats = omats.length
  
  def reset = {
    s1.reset
    s2.reset
  }
  
  def next:Array[Mat] = {
    val mats1 = s1.next;
    val mats2 = s2.next;
    for (i <- 0 until mats1.length) {
      omats(i) = mats1(i);
    }
    for (i <- 0 until mats2.length) {
      omats(i+mats1.length) = mats2(i);
    }
    omats;
  }
  
  def hascol(mats:Array[Mat], iptr:Int, ss:DataSource):Boolean = {
    (iptr < mats(0).ncols) || ss.hasNext
  }
    
  def hasNext:Boolean = {
    s1.hasNext
  }
    
  def progress = {
    s1.progress
  }
}


