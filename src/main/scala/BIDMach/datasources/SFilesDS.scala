package BIDMach.datasources
import BIDMat.{Mat,SBMat,CMat,CSMat,DMat,FMat,IMat,HMat,GMat,GIMat,GSMat,SMat,SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import scala.concurrent.future
import scala.concurrent.ExecutionContext.Implicits.global
import java.io._

/*
 * SFilesDatasource constructs SMat batches from data files stored on disk as IMat. 
 * The IMats are 3-column with column, row indices and integer values.
 * This format allows dynamic construction of the SMat with a specified bound on the max row index,
 * and with specified featurization (e.g. clipped to 1, linear, logarithmic etc.). 
 */

class SFilesDS(override val opts:SFilesDS.Opts = new SFilesDS.Options) extends FilesDS(opts) {
  
  var inptrs:IMat = null
  var offsets:IMat = null
  
  override def init = {
    initbase
    var totsize = sum(opts.fcounts).v
    if (opts.addConstFeat) totsize += 1
    omats = new Array[Mat](1)
    omats(0) = SMat(totsize, opts.batchSize, opts.batchSize * opts.eltsPerSample)
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
    var todo = opts.batchSize
    flushMat(omats(0))
    while (todo > 0 && fileno < opts.nend) {
    	var nrow = rowno
    	val filex = fileno % opts.lookahead
    	while (ready(filex) < fileno) Thread.sleep(1)
    	val spm = spmax(matqueue(filex))
    	nrow = math.min(rowno + todo, spm)
    	val matq = matqueue(filex)
    	if (matq(0) != null) {
    		omats(0) = sprowslice(matq, rowno, nrow, omats(0), opts.batchSize - todo)
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

object SFilesDS {
  trait Opts extends FilesDS.Opts {
  	var fcounts:IMat = null
    var eltsPerSample = 0
  }
  
  class Options extends Opts {}

}

