package BIDMach.models

import BIDMat.{Mat,SBMat,CMat,DMat,FMat,IMat,HMat,GMat,GIMat,GSMat,SMat,SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import BIDMach.datasources._
import BIDMach.updaters._
import BIDMach._

/**
 * An abstract class with shared code for Clustering Models
 */
abstract class ClusteringModel(override val opts:ClusteringModel.Opts) extends Model {
  var lastpos = 0L
  
  def init() = {

    useGPU = opts.useGPU && Mat.hasCUDA > 0
    val data0 = mats(0)
    val m = data0.nrows
    val mmi = rand(opts.dim, m);
    
    setmodelmats(Array[Mat](1));
    modelmats(0) = if (useGPU) GMat(mmi) else mmi
    updatemats = new Array[Mat](1)
    updatemats(0) = modelmats(0).zeros(mmi.nrows, mmi.ncols)
    lastpos = 0;
  } 
  
  def mupdate(data:Mat, ipass:Int):Unit
   
  def evalfun(data:Mat):FMat
  
  def evalfun(data:Mat, targ:Mat):FMat = {col(0)}
  
  def doblock(gmats:Array[Mat], ipass:Int, here:Long) = {
    val mm = modelmats(0);
    val gm = gmats(0);
    if (ipass == 0) {
      if (here.toInt == gm.ncols) {
        println("First pass random centroid initialization")
      }
      val gg = full(gm).t;
      val lastp = lastpos.toInt
      if (lastp < mm.nrows - 1) {
        val step = math.min(gg.nrows, mm.nrows - lastp);
        mm(lastp->(lastp+step),?) = gg(0->step, ?);
//        full(gm).t.rowslice(0, math.min(gm.ncols, mm.nrows - lastp), mm, lastp)
      } else {
        val rp1 = randperm(gm.ncols);
        val rp2 = randperm(mm.nrows);
        val pp = ((here - lastpos) * mm.nrows / here).toInt;
//        println("here %d lastpos %d pp %d" format (here, lastpos,pp))
        if (pp > 0) {
          mm(rp2(0->pp), ?) = gg(rp1(0->pp), ?);        
        }
      }
      lastpos = here;
    } else {
      mupdate(gmats(0), ipass)
    }
  }
  
  def evalblock(mats:Array[Mat], ipass:Int, here:Long):FMat = {
  	lastpos = here;
  	if (mats.length == 1) {
  		evalfun(gmats(0));
  	} else {
  		evalfun(gmats(0), gmats(1));
  	}
  }
}

object ClusteringModel {
  trait Opts extends Model.Opts {
  }
  
  class Options extends Opts {}
}
