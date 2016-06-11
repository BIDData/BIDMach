package BIDMach.datasinks
import BIDMat.{Mat,SBMat,CMat,CSMat,DMat,FMat,IMat,HMat,GMat,GDMat,GIMat,GLMat,GSMat,GSDMat,LMat,SMat,SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import BIDMach.datasources._
import scala.collection.mutable.ListBuffer

class FileSink(override val opts:FileSink.Opts = new FileSink.Options) extends MatSink(opts) { 
  var ifile = 0
  var colsdone = 0
  
  override def init = { 
    blocks = new ListBuffer[Array[Mat]]()
    setnmats(opts.ofnames.length)
    omats = new Array[Mat](nmats)
    ifile = 0
    opts match {
      case fopts:FileSource.Opts => {
        ifile = fopts.nstart
      }
    }
    colsdone = 0
  }
  
  override def put = {
    blocks += omats.map(MatSink.copyCPUmat)
    colsdone += omats(0).ncols
    if (colsdone >= opts.ofcols) {
      mergeSaveBlocks
      colsdone = 0
      ifile += 1
      blocks = new ListBuffer[Array[Mat]]()
    }
  }

  override def close () = {
    mergeSaveBlocks
  }
  
  def mergeSaveBlocks = {
    mergeBlocks
    if (blocks.size > 0) {
      for (i <- 0 until opts.ofnames.length) {
        saveMat(opts.ofnames(i)(ifile), mats(i))
      }
    }
  }
}

object FileSink {
  trait Opts extends MatSink.Opts {
    var ofnames:List[(Int)=>String] = null
    var ofcols = 100000
  }
  
  class Options extends Opts {

  }
}

