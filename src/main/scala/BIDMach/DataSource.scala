package BIDMach
import BIDMat.{Mat,BMat,CMat,CSMat,DMat,FMat,IMat,HMat,GMat,GIMat,GSMat,SMat,SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import scala.actors._
import java.io._

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
  
  def init() = {
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

class FilesDataSource(fnames:CSMat, dirname:(Int)=>String, nstart:Int, nend:Int, 
		opts:FilesDataSource.Options = new FilesDataSource.Options) extends DataSource(opts) { 
  val sizeMargin = opts.sizeMargin
  val blockSize = opts.blockSize
  var fileno = nstart
  var colno = 0
  var omats:Array[Mat] = null
  var matqueue:Array[Array[Mat]] = null
  var ready:IMat = null
  
  def init = {
    fileno = nstart                                           // Number of the current output file
    colno = 0                                                 // Column number in the current output file
    omats = new Array[Mat](fnames.size)
    matqueue = new Array[Array[Mat]](fnames.size)             // Queue of matrices for each output matrix
    ready = -iones(opts.lookahead, 1)                         // Numbers of files currently loaded in queue
    for (i <- 0 until fnames.size) {
      val mm = HMat.loadMat(dirname(nstart) + fnames(i))
      omats(i) = mm match {
      case mm:SMat => SMat(mm.nrows, blockSize, (mm.nnz * sizeMargin * blockSize / mm.ncols).toInt)
      case mm:SDMat => SDMat(mm.nrows, blockSize, (mm.nnz * sizeMargin * blockSize / mm.ncols).toInt)
      case _ => mm.zeros(mm.nrows, blockSize)
      } 
      matqueue(i) = new Array[Mat](opts.lookahead)
    }
    for (i <- 0 until opts.lookahead) {
      Actor.actor {
        prefetch(nstart + i)
      }
    }
  }
  
  def next:Array[Mat] = {
    val filex = fileno % opts.lookahead
    var donextfile = false
    var todo = 0
    var ccols = 0
    while (ready(filex) < fileno) Thread.`yield`
    for (i <- 0 until fnames.size) {
    	val matq = matqueue(i)(filex)
    	ccols = math.min(colno + blockSize, matq.ncols)
    	omats(i) = matq(?, colno -> ccols)
    }
    if (ccols - colno <= blockSize) {
    	todo = blockSize - ccols + colno
    	colno = todo
    	if (todo > 0) {
    		var done = false
    		while (!done && fileno+1 < nend) {
    			val filey = (fileno+1) % opts.lookahead
    			while (ready(filey) < fileno+1) Thread.`yield`
    			if (matqueue(0)(filey) != null) {
    				for (i <- 0 until fnames.size) {
    					val matq = matqueue(i)(filey)
    					omats(i) = omats(i) \ matq(?, 0 -> todo)
    				} 
    				done = true
    			}
    			fileno += 1
    		}
    	} 
    } else {
    	colno += blockSize
    }
    omats
  }
  
  def fileExists(fname:String) = {
    val testme = new File(fname)
    testme.exists
  }
  
  def prefetch(ifile:Int) = {
    val ifilex = ifile % opts.lookahead
  	ready(ifilex) = ifile - opts.lookahead
  	for (inew <- ifile until nend by opts.lookahead) {
  		while (ready(ifilex) >= fileno) Thread.`yield`
  		val fexists = fileExists(dirname(inew) + fnames(0)) && (rand(1,1).v < opts.sampleFiles)
  		for (i <- 0 until fnames.size) {
  			matqueue(i)(ifilex) = if (fexists) HMat.loadMat(dirname(inew) + fnames(i)) else null
  		}
  		ready(ifilex) = inew
  	}
  }
  
  def hasNext:Boolean = {
    (fileno < nend)
  }
}

object FilesDataSource {
  class Options extends DataSource.Options {
  	var lookahead = 8
  	var sampleFiles = 1.0f
  }
  
  def encodeDate(yy:Int, mm:Int, dd:Int) = 372*yy + 31*mm + dd
  
  def decodeDate(n:Int):(Int, Int, Int) = {
    val yy = (n - 32) / 372
    val days = n - 32 - 372 * yy
    val mm = days / 31 + 1
    val dd = days - 31 * (mm - 1) + 1
    (yy, mm, dd)
  }
  
  def sampleFun(middle:String):(Int)=>String = {
    (n:Int) => {    
    	val (yy, mm, dd) = decodeDate(n)
    	("/disk%02d/" + middle + "%04d/%02d/%02d/" format (n % 16, yy, mm, dd))
    }    
  }
}

object DataSource {
  class Options {
    var blockSize = 10000
    var sizeMargin = 5f
  }
  
}
