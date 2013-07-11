package BIDMach
import BIDMat.{Mat,BMat,CMat,CSMat,DMat,FMat,IMat,HMat,GMat,GIMat,GSMat,SMat,SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import scala.actors._
import java.io._

abstract class DataSource(val opts:DataSource.Options = new DataSource.Options) {   
  def next:Array[Mat]  
  def hasNext:Boolean
}

class MatDataSource(mats:Array[Mat], override val opts:DataSource.Options = new DataSource.Options) extends DataSource(opts) { 
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

class FilesDataSource(fnames:List[(Int)=>String], nstart0:Int, nend:Int, transpose:IMat=null,
		override val opts:FilesDataSource.Options = new FilesDataSource.Options) extends DataSource(opts) { 
  var sizeMargin = 0f
  var blockSize = 0 
  var fileno = 0
  var colno = 0
  var omats:Array[Mat] = null
  var matqueue:Array[Array[Mat]] = null
  var ready:IMat = null
  
  def init = {
    var nstart = nstart0
    sizeMargin = opts.sizeMargin
    blockSize = opts.blockSize
    while (!fileExists(fnames(0)(nstart))) {nstart += 1}
    fileno = nstart                                           // Number of the current output file
    colno = 0                                                 // Column number in the current output file
    omats = new Array[Mat](fnames.size)
    matqueue = new Array[Array[Mat]](fnames.size)             // Queue of matrices for each output matrix
    ready = -iones(opts.lookahead, 1)                         // Numbers of files currently loaded in queue
    for (i <- 0 until fnames.size) {
      var mm = HMat.loadMat(fnames(i)(nstart))
      if (transpose.asInstanceOf[AnyRef] != null && transpose(i) == 1) mm = mm.t
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
    while (ready(filex) < fileno) Thread.`yield`
    for (i <- 0 until fnames.size) {
    	val matq = matqueue(i)(filex)
    	if (matq != null) {
    		val ccols = math.min(colno + blockSize, matq.ncols)
    		omats(i) = matq.colslice(colno, ccols, omats(i))
    		todo = blockSize - ccols + colno
    	}
    }
    if (todo > 0) {    	
    	var done = false
    	while (!done && fileno+1 < nend) {
    		val filey = (fileno+1) % opts.lookahead
    		while (ready(filey) < fileno+1) Thread.`yield`
    		if (matqueue(0)(filey).asInstanceOf[AnyRef] != null && matqueue(0)(filey).ncols >= todo) {
    			for (i <- 0 until fnames.size) {
    				val matq = matqueue(i)(filey)
    				omats(i) = omats(i) \ matq(?, 0 -> todo)
    			} 
    			done = true
    		}
    		fileno += 1
    	}
    	colno = todo
    } else {
    	colno += blockSize
    }
    omats
  }
  
  def fileExists(fname:String) = {
    val testme = new File(fname)
    testme.exists
  }
  
  def lazyTranspose(a:Mat) = {
    a match {
      case af:FMat => FMat(a.ncols, a.nrows, af.data)
      case ad:DMat => DMat(a.ncols, a.nrows, ad.data)
      case ai:IMat => IMat(a.ncols, a.nrows, ai.data)
      case _ => throw new RuntimeException("laztTranspose cant deal with "+a.getClass.getName)
    }
  }
  
  def prefetch(ifile:Int) = {
    val ifilex = ifile % opts.lookahead
  	ready(ifilex) = ifile - opts.lookahead
  	for (inew <- ifile until nend by opts.lookahead) {
  		while (ready(ifilex) >= fileno) Thread.`yield`
  		val fexists = fileExists(fnames(0)(inew)) && (rand(1,1).v < opts.sampleFiles)
  		for (i <- 0 until fnames.size) {
  			matqueue(i)(ifilex) = if (fexists) {
  			  val tmp = HMat.loadMat(fnames(i)(inew), true, matqueue(i)(ifilex))
  			  if (transpose.asInstanceOf[AnyRef] != null && transpose(i) == 1) lazyTranspose(tmp) else tmp
  			} else null  			
//  			println("%d" format inew)
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
  
  def encodeDate(yy:Int, mm:Int, dd:Int, hh:Int) = (372*yy + 31*mm + dd)*24 + hh
  
  def decodeDate(n:Int):(Int, Int, Int, Int) = {
    val yy = (n/24 - 32) / 372
    val days = n/24 - 32 - 372 * yy
    val mm = days / 31 + 1
    val dd = days - 31 * (mm - 1) + 1
    val hh = n % 24
    (yy, mm, dd, hh)
  }
  
  def sampleFun(fname:String):(Int)=>String = {
    (n:Int) => {    
    	val (yy, mm, dd, hh) = decodeDate(n)
    	(fname format ((n / 24) % 16, yy, mm, dd, hh))
    }    
  }
  
  def filexx = sampleFun("/disk%02d/twitter/tokenized/%04d/%02d/%02d/tweet%02d.gz")
}

object DataSource {
  class Options {
    var blockSize = 10000
    var sizeMargin = 5f
  }
  
}
