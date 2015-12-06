package BIDMach.datasinks
import BIDMat.{Mat,SBMat,CMat,CSMat,DMat,FMat,IMat,HMat,GMat,GDMat,GIMat,GLMat,GSMat,GSDMat,LMat,SMat,SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import scala.collection.mutable.ListBuffer

class FileSink(override val opts:FileSink.Opts = new FileSink.Options) extends DataSink(opts) { 
  var blocks = new ListBuffer[Array[Mat]]();
  var mats:Array[Mat] = null;
  var ifile = 0;
  var colsdone = 0;
  
  override def init = { 
    blocks = new ListBuffer[Array[Mat]]();
    setnmats(opts.ofnames.length);
    omats = new Array[Mat](nmats);
    ifile = 0;
    colsdone = 0;
  }
  
  def put = {
    blocks += omats.map(MatSink.copyCPUmat);
    colsdone += omats(0).ncols;
    if (colsdone >= opts.ofcols) {
      mergeBlocks;
      colsdone = 0;
      ifile += 1;
      blocks = new ListBuffer[Array[Mat]]();
    }
  }

  override def close () = {
    mergeBlocks;
  }
  
  def mergeBlocks = {
    val ncols = blocks.map(_(0).ncols).reduce(_+_);
    val imats = blocks(0);
    val ablocks = blocks.toArray;
    if (mats == null) mats = new Array[Mat](nmats);
    for (i <- 0 until nmats) {
      val nrows = imats(i).nrows;
      val nnz0 = imats(i) match {
        case i:SMat => i.nnz;
        case i:GSMat => i.nnz;
        case i:SDMat => i.nnz;
        case i:GSDMat => i.nnz;
        case _ => -1;
      }
      mats(i) = if (nnz0 >= 0) {
        val nnz = ablocks.map(_(i).nnz).reduce(_+_);
        SMat(nrows, ncols, nnz);
      } else {
        MatSink.makeCPUmat(imats(i), nrows, ncols);
      }
      var here = 0;
      for (j <- 0 until ablocks.length) {
        val am = ablocks(j)(i);
        am.colslice(0, am.ncols, mats(i), here, true);
        here += am.ncols;
      }
    }
    for (i <- 0 until opts.ofnames.length) {
    	saveMat(opts.ofnames(i)(ifile), mats(i));
    }
  }
}

object FileSink {
  trait Opts extends DataSink.Opts {
  	var ofnames:List[(Int)=>String] = null;
  	var ofcols = 100000;
  }
  
  class Options extends Opts {

  }
}

