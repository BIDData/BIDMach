package BIDMach.datasinks
import BIDMat.{Mat,SBMat,CMat,CSMat,DMat,FMat,IMat,HMat,GMat,GIMat,GSMat,LMat,SMat,SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import scala.collection.mutable.ListBuffer



class MatSink(override val opts:MatSink.Opts = new MatSink.Options) extends DataSink(opts) { 
  var blocks = new ListBuffer[Array[Mat]]();
  var mats:Array[Mat] = null;
  
  override def init = { 
    blocks = new ListBuffer[Array[Mat]]();
  }
  
  def put(m:Array[Mat]) = {
    blocks += m.map(MatSink.copyCPUmat);
  }

  override def close () = {
    val ncols = blocks.map(_(0).ncols).reduce(_+_);
    val imats = blocks(0);
    val ablocks = blocks.toArray;
    val nmats = imats.length;
    mats = new Array[Mat](nmats);
    for (i <- 0 until nmats) {
      val nrows = imats(i).nrows;
      val nnz0 = imats(i) match {
        case i:SMat => i.nnz;
        case i:GSMat => i.nnz;
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
  }
  

}

object MatSink {
    trait Opts extends DataSink.Opts {
  }
  
  class Options extends Opts {   
  }
  
  def copyCPUmat(m:Mat):Mat = {
    val nr = m.nrows;
    val nc = m.ncols;
    val out = makeCPUmat(m, nr, nc);
    out <-- m;
    out;   
  }
  
  def makeCPUmat(m:Mat,nr:Int, nc:Int):Mat = {
  	m match {
  		case f:FMat => zeros(nr,nc);
  		case g:GMat => zeros(nr,nc);
  		case i:IMat => izeros(nr,nc);
  		case gi:GIMat => izeros(nr,nc);
  		case l:LMat => lzeros(nr,nc);
  		case l:LMat => lzeros(nr,nc);
  		case s:SMat => SMat(nr,nc,s.nnz);
  		case s:GSMat => SMat(nr,nc,s.nnz);     
  	}
  }
}

