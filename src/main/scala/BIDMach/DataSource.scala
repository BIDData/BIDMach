package BIDMach
import BIDMat.{Mat,BMat,CMat,CSMat,DMat,FMat,IMat,HMat,GMat,GIMat,GSMat,SMat,SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import scala.actors._
import java.io._

abstract class DataSource(val opts:DataSource.Opts = new DataSource.Options) {   
  def next:Array[Mat]  
  def hasNext:Boolean
  def reset:Unit
  def putBack(mats:Array[Mat],i:Int):Unit = {throw new RuntimeException("putBack not implemented")}
  def setupPutBack(n:Int,dim:Int):Unit = {throw new RuntimeException("putBack not implemented")}
  def nmats:Int
  def init:Unit
  def progress:Float
  var omats:Array[Mat] = null
  var endmats:Array[Mat] = null
  var fullmats:Array[Mat] = null
}

class MatDataSource(var mats:Array[Mat], override val opts:MatDataSource.Opts = new MatDataSource.Options) extends DataSource(opts) { 
  var sizeMargin = 0f 
  var here = 0
  var there = 0
  var blockSize = 0
  var totalSize = 0
  var umat:Mat = null
  
  def init = {
    sizeMargin = opts.sizeMargin
    blockSize = opts.blockSize
    if (opts.addConstFeat) {
      mats(0) = mats(0) on sparse(ones(1, mats(0).ncols)) 
    }
    if (opts.featType == 0) {
      mats(0).contents.set(1)     
    }
    here = -blockSize
    totalSize = mats(0).ncols
    omats = new Array[Mat](mats.length)
    endmats = new Array[Mat](mats.length)
    fullmats = new Array[Mat](mats.length)   
  }
  
  def nmats = omats.length
  
  def reset = {
    here = -blockSize
  }
  
  def next:Array[Mat] = {
    here = math.min(here+blockSize, mats(0).ncols)
    there = math.min(here+blockSize, mats(0).ncols)
  	for (i <- 0 until mats.length) {
  	  if (there - here == blockSize) {
  	    fullmats(i) = mats(i).colslice(here, there, fullmats(i))
  	    omats(i) = fullmats(i)
  	  } else {
  	    endmats(i) = mats(i).colslice(here, there, endmats(i))
  	    omats(i) = endmats(i) 	    
  	  }
  	}
  	omats
  }
  
  def hasNext:Boolean = {
    here + blockSize < mats(0).ncols
  }
  
  override def setupPutBack(n:Int, dim:Int) = {
    if (mats.length < n || mats(n-1).asInstanceOf[AnyRef] == null || mats(n-1).nrows != dim) {
      val newmats = new Array[Mat](n)
      for (i <- 0 until n-1) {
        newmats(i) = mats(i)
      }
      newmats(n-1) = ones(dim, mats(0).ncols)
      mats = newmats
    } 
  }
  
  override def putBack(tmats:Array[Mat],i:Int):Unit = {
    tmats(i).colslice(0, tmats(i).ncols, mats(i), here)
  }
  
  def progress = {
    math.min((here+blockSize)*1f/totalSize, 1f)
  }

}

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

class SFilesDataSource(override val opts:SFilesDataSource.Opts = new SFilesDataSource.Options) extends FilesDataSource(opts) {
  
  var inptrs:IMat = null
  var offsets:IMat = null
  
  override def init = {
    initbase
    var totsize = sum(opts.fcounts).v
    if (opts.addConstFeat) totsize += 1
    omats = new Array[Mat](1)
    omats(0) = SMat(totsize, opts.blockSize, opts.blockSize * opts.eltsPerSample)
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
    val addConstFeat = opts.addConstFeat
    val featType = opts.featType
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
        val mrows = mat.nrows
        var k = inptrs(j)
        while (k < mrows && mat.data(k) < irow) k += 1
        inptrs(j) = k
        val xoff = innz - k
        val yoff = offsets(j) + ioff
        while (k < mat.nrows && mat.data(k) == irow && mat.data(k+mrows) < lims(j)) {
          omat.ir(xoff + k) = mat.data(k+mrows) + yoff
          omat.data(xoff + k) = if (featType == 0) 1f else mat.data(k+2*mrows)
          k += 1
        }
        innz = xoff + k
        inptrs(j) = k
        j += 1
      }
      irow += 1
      idone += 1
      if (addConstFeat) {
        omat.ir(innz) = omat.nrows - 1 + ioff
        omat.data(innz) = 1
        innz += 1
      }
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
    	while (ready(filex) < fileno) Thread.sleep(1)
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

class BlendedDataSource(val s1:DataSource, val s2:DataSource, var alpha:Float, var samp1:Float, var samp2:Float,
    override val opts:BlendedDataSource.Opts = new BlendedDataSource.Options) extends DataSource(opts) {
  var sizeMargin = 0f 
  var here = 0L
  var there = 0
  var iptr1 = 0
  var iptr2 = 0
  var blockSize = 0
  var bBlock = 0
  var totalSize = 0
  var randv:FMat = null
  var rands1:FMat = null
  var rands2:FMat = null
  var mats1:Array[Mat] = null
  var mats2:Array[Mat] = null
  omats = null
  
  def init = {
    sizeMargin = opts.sizeMargin
    blockSize = opts.blockSize
    bBlock = opts.bBlock
    randv = rand(1, blockSize/bBlock + 1)
    rands1 = rand(1, blockSize/bBlock + 1)
    rands2 = rand(1, blockSize/bBlock + 1)
    here = -blockSize
    s1.opts.addConstFeat = opts.addConstFeat
    s2.opts.addConstFeat = opts.addConstFeat
    s1.opts.featType = opts.featType
    s2.opts.featType = opts.featType
    s1.init
    s2.init
    mats1 = s1.next
    mats2 = s2.next
    totalSize = mats1(0).ncols
    omats = new Array[Mat](mats1.length)
    for (i <- 0 until mats1.length) {
      omats(i) = mats1(i) match {
        case mm:SMat => SMat(mats1(i).nrows, blockSize, (mats1(i).nnz * sizeMargin).toInt)
        case mm:SDMat => SDMat(mats1(i).nrows, blockSize, (mats1(i).nnz * sizeMargin).toInt)
        case _ => mats1(i).zeros(mats1(i).nrows, blockSize)
      }      
    }    
  }
  
  def nmats = omats.length
  
  def reset = {
    s1.reset
    s2.reset
    here = -blockSize
  }
  
  @inline def copycol(inmats:Array[Mat], iptr:Int, jptr:Int, omats:Array[Mat], here:Int) = {
    var imat = 0
    while (imat < inmats.length) {
      omats(imat) = inmats(imat).colslice(iptr, jptr, omats(imat), here)
      imat += 1
    }
  }
  
  def next:Array[Mat] = {
    rand(0, 1f, randv)
    var i = 0
    var xptr = 0
    while (xptr < blockSize && hascol(mats1, iptr1, s1) && hascol(mats2, iptr2, s2)) {
      if (randv.data(i) < alpha) {
        while (iptr1 < mats1(0).ncols && rands1.data(iptr1/bBlock) > samp1) iptr1 += bBlock
        if (iptr1 >= mats1(0).ncols) {
          mats1 = s1.next
          iptr1 = 0
          rand(0, 1f, samp1)
        }
        val jptr1 = math.min(mats1(0).ncols, iptr1 + math.min(bBlock, math.min(blockSize, omats(0).ncols) - xptr))
        copycol(mats1, iptr1, jptr1,  omats, xptr)
        xptr += jptr1 - iptr1
        iptr1 = jptr1
      } else {
        while (iptr2 < mats2(0).ncols && rands2.data(iptr2/bBlock) > samp2) iptr2 += bBlock
      	if (iptr2 >= mats2(0).ncols) {
          mats2 = s2.next
          iptr2 = 0
          rand(0, 1f, samp2)
        }
        val jptr2 = math.min(mats1(0).ncols, iptr2 + math.min(bBlock, math.min(blockSize, omats(0).ncols) - xptr))
        copycol(mats1, iptr2, jptr2,  omats, xptr)
        xptr += jptr2 - iptr2
        iptr2 = jptr2
      }
      i += 1
    }
    here += xptr
    if (xptr == blockSize) {
      omats
    } else {
      shrinkmats(omats, i)
    }
  }
  
  def hascol(mats:Array[Mat], iptr:Int, ss:DataSource):Boolean = {
    (iptr < mats(0).ncols) || ss.hasNext
  }
    
  def hasNext:Boolean = {
    hascol(mats1, iptr1, s1) && hascol(mats2, iptr2, s2)
  }
  
  def shrinkmats(xmats:Array[Mat], n:Int) = {
    val outarr = new Array[Mat](omats.length)
    var imat = 0
    while (imat < omats.length) {
      outarr(imat) = xmats(imat).colslice(0, n, null)
      imat += 1
    }
    outarr
  }
    
  def progress = {
    math.max(s1.progress, s2.progress)
  }
}


object DataSource {
  trait Opts {
    var blockSize = 100000
    var sizeMargin = 3f
    var sample = 1f
    var addConstFeat:Boolean = false
    var featType:Int = 1                 // 0 = binary features, 1 = linear features
  } 
  
  class Options extends Opts {}
}

object MatDataSource {
  trait Opts extends DataSource.Opts {
  }
  
  class Options extends Opts {   
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

object BlendedDataSource {
  trait Opts extends DataSource.Opts {
  	var bBlock = 1000
  }
  
  class Options extends Opts {}
}

object SFilesDataSource {
  trait Opts extends FilesDataSource.Opts {
  	var fcounts:IMat = null
    var eltsPerSample = 0
  }
  
  class Options extends Opts {}
  
  val twitterFeatureDir = "/disk%02d/twitter/featurized/%04d/%02d/%02d/"
  val twitterSmileyFeatureDir = "/disk%02d/twitter/smiley/featurized/%04d/%02d/%02d/"
  
  def twitterWords(
      nstart0:Int = FilesDataSource.encodeDate(2012,3,1,0),
  		nend0:Int = FilesDataSource.encodeDate(2012,12,1,0),
  		n:Int = 1,
  		i:Int = 0,
  		nfeats:Int = 100000) = {
  	val opts = new SFilesDataSource.Options { 
  		override def fnames:List[(Int)=>String] = List(FilesDataSource.sampleFun(twitterFeatureDir + "unifeats%02d.lz4", n, i))
  		fcounts = icol(nfeats)
  		nstart = nstart0/n
  		nend = nend0/n
  		order = 1
  		blockSize = 100000
  		eltsPerSample = 40
  		lookahead = 3
  	}
  	new SFilesDataSource(opts)
  }
  
  def twitterSmileyWords(
  		nstart0:Int = FilesDataSource.encodeDate(2012,3,1,0),
  		nend0:Int = FilesDataSource.encodeDate(2013,7,1,0),
  		n:Int = 1,
  		i:Int = 0,
  		nfeats:Int = 100000) = {
  	val opts = new SFilesDataSource.Options { 
  		override def fnames:List[(Int)=>String] = List(FilesDataSource.sampleFun(twitterSmileyFeatureDir + "unifeats%02d.lz4", n, i))
  		fcounts = icol(nfeats)
  		nstart = nstart0/n
  		nend = nend0/n
  		order = 1
  		blockSize = 100000
  		eltsPerSample = 40
  		lookahead = 3
  	}
  	new SFilesDataSource(opts)
  }
  
  def twitterNgrams(
      nstart0:Int = FilesDataSource.encodeDate(2012,3,1,0),
  		nend0:Int = FilesDataSource.encodeDate(2012,12,1,0),
  		n:Int = 1,
  		i:Int = 0,
  		nuni0:Int = 50,
  		nbi0:Int = 100,
  		ntri0:Int = 200) = {
  	val opts = new SFilesDataSource.Options { 
  		override def fnames:List[(Int)=>String] = List(
  				FilesDataSource.sampleFun(twitterFeatureDir + "unifeats%02d.lz4", n, i),
  				FilesDataSource.sampleFun(twitterFeatureDir + "bifeats%02d.lz4", n, i),
  				FilesDataSource.sampleFun(twitterFeatureDir + "trifeats%02d.lz4", n, i)
  		    )
  		fcounts = icol(nuni0*1000,nbi0*1000,ntri0*1000)
  		nstart = nstart0/n
  		nend = nend0/n
  		order = 1
  		blockSize = 100000
  		eltsPerSample = 40
  		lookahead = 3
  	}
  	new SFilesDataSource(opts)
  }
  
  def twitterSmileyNgrams(
      nstart0:Int = FilesDataSource.encodeDate(2012,3,1,0),
  		nend0:Int = FilesDataSource.encodeDate(2013,7,1,0),
  		n:Int = 1,
  		i:Int = 0,
  		nuni0:Int = 50,
  		nbi0:Int = 100,
  		ntri0:Int = 200) = {
  	val opts = new SFilesDataSource.Options { 
  		override def fnames:List[(Int)=>String] = List(
  				FilesDataSource.sampleFun(twitterSmileyFeatureDir + "unifeats%02d.lz4", n, i),
  				FilesDataSource.sampleFun(twitterSmileyFeatureDir + "bifeats%02d.lz4", n, i),
  				FilesDataSource.sampleFun(twitterSmileyFeatureDir + "trifeats%02d.lz4", n, i)
  		    )
  		fcounts = icol(nuni0*1000,nbi0*1000,ntri0*1000)
  		nstart = nstart0/n
  		nend = nend0/n 
  		order = 1
  		blockSize = 100000
  		eltsPerSample = 40
  		lookahead = 3
  	}
  	new SFilesDataSource(opts)
  }
   
  def twitterWordBlend(
  		nstart0:Int = FilesDataSource.encodeDate(2012,3,1,0),
  		nend0:Int = FilesDataSource.encodeDate(2013,7,1,0),
  		n:Int = 1,
  		i:Int = 0,
  		nfeats:Int = 10000) = {  
    val ds1 = twitterWords(nstart0, nend0, n, i, nfeats)
    val ds2 = twitterSmileyWords(nstart0, nend0, n, i, nfeats)
    if (n > 1) {
    	ds1.opts.lookahead = 2
    	ds2.opts.lookahead = 2
    }
    val opts3 = new BlendedDataSource.Options
    new BlendedDataSource(ds1, ds2, 0.5f, 1f, 1f, opts3)
  }
  
  def twitterNgramBlend( 
  		nstart0:Int = FilesDataSource.encodeDate(2012,3,1,0),
  		nend0:Int = FilesDataSource.encodeDate(2013,7,1,0),
  		n:Int = 1,
  		i:Int = 0,
  		nuni0:Int = 50,
  		nbi0:Int = 100,
  		ntri0:Int = 200) = {
    val ds1 = twitterNgrams(nstart0, nend0, n, i, nuni0, nbi0, ntri0)
    val ds2 = twitterSmileyNgrams(nstart0, nend0, n, i, nuni0, nbi0, ntri0)
    if (n > 1) {
    	ds1.opts.lookahead = 2
    	ds2.opts.lookahead = 2
    }
    val opts3 = new BlendedDataSource.Options
    new BlendedDataSource(ds1, ds2, 0.7f, 1f, 1f, opts3)
  }
  
  def testSources(nthreads:Int=4,ff:(Int,Int,Int,Int,Int)=>DataSource = twitterWords, nfeats:Int=100000):IMat = { 
  	val nstart0 = FilesDataSource.encodeDate(2012,3,22,0)
    val nend0 = FilesDataSource.encodeDate(2013,7,1,0)
    var bytes = 0L
    var done = 0L
    var step = 10000000000L
    var stop = izeros(1,1)
    tic
    for (i <- 0 until nthreads) { 
      scala.actors.Actor.actor { 
        val ss = ff(nstart0, nend0, nthreads, i, nfeats)
        ss.init
        while (ss.hasNext && stop.v != 1) { 
        	val a = ss.next
        	bytes += 12L*a(0).nnz
        	if (bytes > done + step) { 
        		done = (bytes/step)*step
        		val t=toc
        		println("GB=%4.2f, t=%4.2f, MB/s=%4.2f" format (bytes/1e9, t, bytes/t/1e6))
        	}
        }
        val t = toc
        println("Thread %d done, GB=%4.2f, t=%4.2f, MB/s=%4.2f" format (i, bytes/1e9, t, bytes/t/1e6))
      }
    }
  	stop
  }
}

