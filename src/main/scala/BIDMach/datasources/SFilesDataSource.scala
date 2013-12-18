package BIDMach.datasources
import BIDMat.{Mat,BMat,CMat,CSMat,DMat,FMat,IMat,HMat,GMat,GIMat,GSMat,SMat,SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import scala.actors._
import java.io._


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

