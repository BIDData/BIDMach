package BIDMach
import BIDMat.{Mat,BMat,CMat,CSMat,DMat,FMat,IMat,HMat,GMat,GIMat,GSMat,SMat,SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._

abstract class DataSource(opts:DataSource.Options = new DataSource.Options) { 
  val options = opts
  
  def next:Array[Mat]
  
  def hasNext:Boolean
}

class MatDataSource(mats:Array[Mat], opts:DataSource.Options = new DataSource.Options) extends DataSource(opts) { 
  val sizeMargin = opts.sizeMargin
  var here = 0
  val blockSize = opts.blockSize
  var omats:Array[Mat] = null
  
  def init = {
    here = 0
    omats = new Array[Mat](mats.length)
    for (i <- 0 until mats.length) {
      omats(i) = mats(i) match {
        case mm:SMat => SMat(mats(i).nrows, blockSize, (mats(i).nnz * sizeMargin * blockSize / mats(i).ncols).toInt)
        case mm:SDMat => SDMat(mats(i).nrows, blockSize, (mats(i).nnz * sizeMargin * blockSize / mats(i).ncols).toInt)
        case _ => mats(i).zeros(mats(i).nrows, blockSize)
      }      
    }    
  }
  
  def next:Array[Mat] = {
    val there = math.min(here+blockSize, mats(0).ncols)
  	for (i <- 0 until mats.length) {
  	  omats(i) = mats(i).colslice(here, there, omats(i))
  	}
    here = math.min(here+blockSize, mats(0).ncols)
  	omats
  }
  
  def hasNext:Boolean = {
    here < mats(0).ncols
  }

}

class FilesDataSource(fnames:Array[CSMat], opts:FilesDataSource.Options = new FilesDataSource.Options) extends DataSource(opts) { 
  val sizeMargin = opts.sizeMargin
  val blockSize = opts.blockSize
  var fileno = 0
  var colno = 0
  var omats:Array[Mat] = null
  var matqueue:Array[Array[Mat]] = null
  var ready:IMat = null
  
  def init = {
    fileno = 0
    colno = 0
    omats = new Array[Mat](fnames.size)
    matqueue = new Array[Array[Mat]](fnames.size)
    ready = IMat(fnames.size, opts.lookahead)
    for (i <- 0 until fnames.size) {
      val mm = HMat.loadMat(fnames(i)(0))
      omats(i) = mm match {
      case mm:SMat => SMat(mm.nrows, blockSize, (mm.nnz * sizeMargin * blockSize / mm.ncols).toInt)
      case mm:SDMat => SDMat(mm.nrows, blockSize, (mm.nnz * sizeMargin * blockSize / mm.ncols).toInt)
      case _ => mm.zeros(mm.nrows, blockSize)
      } 
      matqueue(i) = new Array[Mat](opts.lookahead)
    }
  }
  
  def next:Array[Mat] = {
    val filex = fileno % opts.lookahead
    var donextfile = false
    var todo = 0
    for (i <- 0 until fnames.size) {
    	while (ready(i, filex) == 0) Thread.`yield`
    	val matq = matqueue(i)(filex)
    	val ccols = math.min(colno + blockSize, matq.ncols)
    	omats(i)(?, 0 -> (ccols - colno)) = matq(?, colno -> ccols)
    	if (ccols - colno < blockSize) {
    	  ready(i, filex) = 0
    	  todo = blockSize - ccols + colno
    	  donextfile = true

    	  if (fileno+1 < fnames(0).length) {
    	    val filey = filex + 1
    	    val matq = matqueue(i)(filey)
    	    while (ready(i, filey) == 0) Thread.`yield`
    	    omats(i)(?, (ccols - colno) -> blockSize) = matq(?, 0 -> todo)
    	  }
    	}
    }
    if (donextfile) {
      colno = todo
    } else {
    	colno += blockSize
    }
    omats
  }
  
  def hasNext:Boolean = {
    true
  }
}

object FilesDataSource {
  class Options extends DataSource.Options {
  	var lookahead = 8
  }
  
  def dateid(yy:Int, mm:Int, dd:Int) = 372*yy + 31*mm + dd
}

object DataSource {
  class Options {
    var blockSize = 10000
    var sizeMargin = 5f
  }
  
}
