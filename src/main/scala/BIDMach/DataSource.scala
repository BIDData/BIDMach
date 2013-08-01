package BIDMach
import BIDMat.{Mat,BMat,CMat,CSMat,DMat,FMat,IMat,HMat,GMat,GIMat,GSMat,SMat,SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import scala.actors._
import java.io._

abstract class DataSource(val opts:DataSource.Options = new DataSource.Options) {   
  def next:Array[Mat]  
  def hasNext:Boolean
  def reset:Unit
  def putBack(mats:Array[Mat],i:Int)
}

class MatDataSource(val mats:Array[Mat], override val opts:DataSource.Options = new DataSource.Options) extends DataSource(opts) { 
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
  
  def reset = {
    here = 0
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
  
  def putBack(mats:Array[Mat],i:Int) = {}

}

class FilesDataSource(override val opts:FilesDataSource.Options = new FilesDataSource.Options) extends DataSource(opts) { 
  var sizeMargin = 0f
  var blockSize = 0 
  var fileno = 0
  var rowno = 0
  var nstart = 0
  var fnames:List[(Int)=>String] = null
  var omats:Array[Mat] = null
  var matqueue:Array[Array[Mat]] = null
  var ready:IMat = null
  var stop:Boolean = false
  
  def initbase = {
    nstart = opts.nstart
    fnames = opts.fnames
    blockSize = opts.blockSize
    while (!fileExists(fnames(0)(nstart))) {nstart += 1}
    fileno = nstart                                                    // Number of the current output file
    rowno = 0                                                          // row number in the current output file
    matqueue = new Array[Array[Mat]](opts.lookahead)                   // Queue of matrices for each output matrix
    ready = -iones(opts.lookahead, 1)                                  // Numbers of files currently loaded in queue
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
  
  def next:Array[Mat] = {
    var donextfile = false
    var todo = blockSize
    while (todo > 0 && fileno < opts.nend) {
    	var nrow = rowno
    	val filex = fileno % opts.lookahead
    	while (ready(filex) < fileno) Thread.`yield`
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
  		while (ready(ifilex) >= fileno) Thread.`yield`
  		val inew = ready(ifilex) + opts.lookahead
  		val fexists = fileExists(fnames(0)(inew)) && (rand(1,1).v < opts.sampleFiles)
  		for (i <- 0 until fnames.size) {
  			matqueue(ifilex)(i) = if (fexists) {
  			  HMat.loadMat(fnames(i)(inew), matqueue(ifilex)(i))  			 
  			} else null  			
//  			println("%d" format inew)
  		}
  		ready(ifilex) = inew
  	}
  }
  
  def hasNext:Boolean = {
    (fileno < opts.nend)
  }
  
  def putBack(mats:Array[Mat],i:Int) = {}
}

class SFilesDataSource(override val opts:SFilesDataSource.Options = new SFilesDataSource.Options) extends FilesDataSource(opts) {
  
  var inptrs:IMat = null
  var offsets:IMat = null
  
  override def init = {
    initbase
    var totsize = sum(opts.fcounts).v
    omats = new Array[Mat](1)
    omats(0) = SMat(totsize, opts.blockSize, opts.sBlockSize)
    inptrs = izeros(opts.fcounts.length, 1)
    offsets = 0 on cumsum(opts.fcounts)
  }
  
  def binFind(i:Int, mat:Mat):Int = {
    val imat = mat.asInstanceOf[IMat]
    val nrows = mat.nrows
    var ibeg = 0
    var iend = nrows
    while (ibeg < iend) {
      val imid = (iend + ibeg)/2
      if (i > imat(imid, 0)) {
        ibeg = imid+1
      } else {
        iend = imid
      }
    }
    iend    
  }
  
  def sprowslice(inmat:Array[Mat], rowno:Int, nrow:Int, omat0:Mat, done:Int):Mat = {
    val omat = omat0.asInstanceOf[SMat]
    val ioff = Mat.ioneBased
    var idone = done
    var innz = omat.nnz
    val lims = opts.fcounts
    val nfiles = opts.fcounts.length
    var j = 0
    while (j < nfiles) {
    	inptrs(j, 0) = binFind(rowno, inmat(j))
    	j += 1
    }
    var irow = rowno
    while (irow < nrow) {
      var j = 0
      while (j < nfiles) {
        val mat = inmat(j).asInstanceOf[IMat]
        var k = inptrs(j)
        while (k < mat.nrows && mat(k, 0) < irow) k += 1
        inptrs(j) = k
        val xoff = innz - k
        val yoff = offsets(j) + ioff
        while (k < mat.nrows && mat(k, 0) == irow && mat(k, 1) < lims(j)) {
          omat.ir(xoff + k) = mat(k, 1) + yoff
          omat.data(xoff + k) = mat(k, 2)
          k += 1
        }
        innz = xoff + k
        inptrs(j) = k
        j += 1
      }
      irow += 1
      idone += 1
      omat.jc(idone) = innz + ioff
    }
    omat.nnz0 = innz
    omat    
  }
  
  def spmax(matq:Array[Mat]):Int = {
    var maxv = 0
    for (i <- 0 until matq.length) {
      if (matq(i) != null) {
      	val mat = matq(i).asInstanceOf[IMat]
      	maxv = math.max(maxv, mat(mat.nrows-1,0))
      }
    }
    maxv
  }
  
  def fillup(mat:Mat, todo:Int) = {
    val smat = mat.asInstanceOf[SMat]
    val ncols = mat.ncols
    var i = ncols - todo
    val theend = smat.jc(i)
    while (i < ncols) {
      i += 1
      smat.jc(i) = theend
    }
  }
  
  def flushMat(mat:Mat) = {
    val smat = mat.asInstanceOf[SMat]
    smat.nnz0 = 0
    smat.jc(0) = Mat.ioneBased
  }
  
  override def next:Array[Mat] = {
    var donextfile = false
    var todo = blockSize
    flushMat(omats(0))
    while (todo > 0 && fileno < opts.nend) {
    	var nrow = rowno
    	val filex = fileno % opts.lookahead
    	while (ready(filex) < fileno) Thread.`yield`
    	val spm = spmax(matqueue(filex))
    	nrow = math.min(rowno + todo, spm)
    	val matq = matqueue(filex)
    	if (matq(0) != null) {
    		omats(0) = sprowslice(matq, rowno, nrow, omats(0), blockSize - todo)
    		if (spm == nrow) donextfile = true
    	} else {
    		donextfile = true
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
    if (todo > 0) {
      fillup(omats(0), todo)
    }
    omats
  }

}

object FilesDataSource {
  
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
  
  class Options extends DataSource.Options {
  	val localDir = "/disk%02d/twitter/featurized/%04d/%02d/%02d/"
  	def fnames:List[(Int)=>String] = List(sampleFun(localDir + "unifeats%02d.lz4"),
  			                                  sampleFun(localDir + "bifeats%02d.lz4"),
  			                                  sampleFun(localDir + "trifeats%02d.lz4"))
  	var lookahead = 8
  	var sampleFiles = 1.0f
    var nstart:Int = encodeDate(2011,11,22,0)
    var nend:Int = encodeDate(2013,6,31,0)
    var dorows:Boolean = true
  }
}

object SFilesDataSource {
  class Options extends FilesDataSource.Options {
  	override val localDir = "/disk%02d/twitter/featurized/%04d/%02d/%02d/"
  	override def fnames:List[(Int)=>String] = List(FilesDataSource.sampleFun(localDir + "unifeats%02d.lz4"),
  			                                           FilesDataSource.sampleFun(localDir + "bifeats%02d.lz4"),
  			                                           FilesDataSource.sampleFun(localDir + "trifeats%02d.lz4"))
  	var fcounts = icol(10000,20000,100000)
  	lookahead = 8
  	sampleFiles = 1.0f
    nstart = FilesDataSource.encodeDate(2011,11,22,0)
    nend = FilesDataSource.encodeDate(2013,6,31,0)
    blockSize = 10000
    var sBlockSize = 500000
  }
  
  val singleOpts = new Options {
    override def fnames:List[(Int)=>String] = List(FilesDataSource.sampleFun(localDir + "unifeats%02d.lz4"))
    fcounts = icol(100000)
  }
}

object DataSource {
  class Options {
    var blockSize = 100000
    var nusers  = 1000000L
    var sizeMargin = 5f
  }
  
}
