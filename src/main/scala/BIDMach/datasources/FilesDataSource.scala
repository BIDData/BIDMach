package BIDMach.datasources
import BIDMat.{Mat,BMat,CMat,CSMat,DMat,FMat,IMat,HMat,GMat,GIMat,GSMat,SMat,SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import scala.actors._
import java.io._

class FilesDataSource(override val opts:FilesDataSource.Opts = new FilesDataSource.Options) extends DataSource(opts) { 
  var sizeMargin = 0f
  var blockSize = 0 
  @volatile var fileno = 0
  var rowno = 0
  var nstart = 0
  var fnames:List[(Int)=>String] = null
  omats = null
  var matqueue:Array[Array[Mat]] = null
  var ready:IMat = null
  var stop:Boolean = false
  var permfn:(Int)=>Int = null
  var totalSize = 0
  
  def softperm(nstart:Int, nend:Int) = {
    val dd1 = nstart / 24
    val hh1 = nstart % 24
    val dd2 = nend / 24
    val hh2 = nend % 24
    val (dmy, ii) = sort2(rand(dd2-dd1+1+opts.lookahead))
    (n:Int) => {
    	val dd = n / 24
    	val hh = n % 24
    	val ddx = ii(dd-dd1)+dd1
    	val ddx0 = ddx % 31
    	val ddx1 = ddx / 31
    	val hhdd = hh + 24 * (ddx0 - 1)
    	(ddx1 * 31 + (hhdd % 31 + 1)) * 24 + hhdd / 31
    }    
  }
  
  def initbase = {
    nstart = opts.nstart
    fnames = opts.fnames
    blockSize = opts.blockSize
    while (!fileExists(fnames(0)(nstart))) {nstart += 1}
    if (opts.order == 1) {
    	val (dmy, rr) = sort2(rand(opts.nend+opts.lookahead+1-nstart,1))         // Randomize the file read order
    	permfn = (a:Int) => rr(a-nstart)+nstart
    } else {
      permfn = (n:Int) => {                                                    // Stripe reads across disks (different days)
        val (yy, mm, dd, hh) = FilesDataSource.decodeDate(n)
        val hhdd = hh + 24 * (dd - 1)
        FilesDataSource.encodeDate(yy, mm, hhdd % 31 + 1, hhdd / 31)
      } 
    }    
    fileno = nstart                                                            // Number of the current output file
    rowno = 0                                                                  // row number in the current output file
    totalSize = opts.nend - nstart
    matqueue = new Array[Array[Mat]](opts.lookahead)                           // Queue of matrices for each output matrix
    ready = -iones(opts.lookahead, 1)                                          // Numbers of files currently loaded in queue
    for (i <- 0 until opts.lookahead) {
      matqueue(i) = new Array[Mat](fnames.size)
    }
    for (i <- 0 until opts.lookahead) {
      Actor.actor {
        prefetch(nstart + i)
      }
    }
  }
  
  def reset = {
    fileno = nstart
    rowno = 0
    for (i <- 0 until opts.lookahead) {
      val ifile = nstart + i
      val ifilex = ifile % opts.lookahead
      ready(ifilex) = ifile - opts.lookahead
    } 
  }
  
  def init = {
    initbase
    omats = new Array[Mat](fnames.size)
    for (i <- 0 until fnames.size) {
      var mm = HMat.loadMat(fnames(i)(nstart))
      if (opts.dorows) {
      	omats(i) = mm.zeros(blockSize, mm.ncols)
      } else {
      	omats(i) = mm.zeros(mm.nrows, blockSize)
      }
    } 
  }
  
  def progress = {
    (fileno-nstart)*1f / totalSize
  }
  
  def nmats = omats.length
  
  def next:Array[Mat] = {
    var donextfile = false
    var todo = blockSize
    while (todo > 0 && fileno < opts.nend) {
    	var nrow = rowno
    	val filex = fileno % opts.lookahead
    	while (ready(filex) < fileno) Thread.sleep(1)
    	for (i <- 0 until fnames.size) {
    		val matq = matqueue(filex)(i)
    		if (matq != null) {
    		  val matqnr = if (opts.dorows) matq.nrows else matq.ncols
    			nrow = math.min(rowno + todo, matqnr)
    			if (opts.dorows) {
      			omats(i) = matq.rowslice(rowno, nrow, omats(i), blockSize - todo)  			  
    			} else {
    				omats(i) = matq.colslice(rowno, nrow, omats(i), blockSize - todo)   			  
    			}
    			if (matqnr == nrow) donextfile = true
    		} else {
    		  donextfile = true
    		}
    	}
    	todo -= nrow - rowno
    	if (donextfile) {
    	  fileno += 1
    	  rowno = 0
    	  donextfile = false
    	} else {
    		rowno = nrow
    	}
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
  	while  (!stop) {
  		while (ready(ifilex) >= fileno) Thread.sleep(1)
  		val inew = ready(ifilex) + opts.lookahead
  		val pnew = permfn(inew)
  		val fexists = fileExists(fnames(0)(pnew)) && (rand(1,1).v < opts.sampleFiles)
  		for (i <- 0 until fnames.size) {
  			matqueue(ifilex)(i) = if (fexists) {
  			  HMat.loadMat(fnames(i)(pnew), matqueue(ifilex)(i))  			 
  			} else null  			
//  			println("%d" format inew)
  		}
  		ready(ifilex) = inew
  	}
  }
  
  def hasNext:Boolean = {
    (fileno < opts.nend)
  }

}


object FilesDataSource {
  
  def encodeDate(yy:Int, mm:Int, dd:Int, hh:Int) = (((12*yy + mm) * 31) + dd)*24 + hh
  
  def decodeDate(n:Int):(Int, Int, Int, Int) = {
    val days = n / 24
    val dd = (days - 1) % 31 + 1
    val months = (days - dd) / 31
    val mm = (months - 1) % 12 + 1
    val yy = (months - mm) / 12
    (yy, mm, dd, n % 24)
  }
  
  def sampleFun(fname:String):(Int)=>String = {
    (n:Int) => {    
    	val (yy, mm, dd, hh) = decodeDate(n)
    	(fname format ((n / 24) % 16, yy, mm, dd, hh))
    }    
  }
  
  def sampleFun(fname:String, m:Int, i:Int):(Int)=>String = {
    (n0:Int) => { 
      val n = n0 * m + i
    	val (yy, mm, dd, hh) = decodeDate(n)
    	(fname format ((n / 24) % 16, yy, mm, dd, hh))
    }    
  }
  
 
  trait Opts extends DataSource.Opts {
  	val localDir:String = ""
  	def fnames:List[(Int)=>String] = null
  	var lookahead = 8
  	var sampleFiles = 1.0f
    var nstart:Int = 0
    var nend:Int = 0
    var dorows:Boolean = true
    var order:Int = 1                           // 0 = sequential order, 1 = random
  }
  
  class Options extends Opts {}
}
