package BIDMach.models

import BIDMat.{Mat,SBMat,CMat,DMat,FMat,FND,IMat,HMat,GMat,GIMat,GSMat,GND,ND,SMat,SDMat}
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
    if (refresh) {
    	val mmi = rand(opts.dim, m);   
    	setmodelmats(Array(mmi));
    }
    modelmats(0) = convertMat(modelmats(0))
    updatemats = new Array[ND](1)
    updatemats(0) = modelmats(0).zeros(modelmats(0).nrows, modelmats(0).ncols)
    lastpos = 0;
  } 
  
  def mupdate(data:Mat, ipass:Int):Unit
   
  def evalfun(data:Mat):FMat
  
  def evalfun(data:Mat, targ:Mat):FMat = {col(0)}
  
  def dobatch(gmats:Array[ND], ipass:Int, here:Long) = {
    val mm = modelmats(0).asMat;
    val gm = gmats(0).asMat;
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
      mupdate(gmats(0).asMat, ipass)
    }
  }
  
  def evalbatch(mats:Array[ND], ipass:Int, here:Long):FMat = {
  	lastpos = here;
  	if (mats.length == 1) {
  		evalfun(gmats(0).asMat);
  	} else {
  		evalfun(gmats(0).asMat, gmats(1).asMat);
  	}
  }
}

object ClusteringModel {
  trait Opts extends Model.Opts {
  }
  
  class Options extends Opts {}
}
