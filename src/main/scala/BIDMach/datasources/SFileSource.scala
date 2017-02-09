package BIDMach.datasources
import BIDMat.{Mat,SBMat,CMat,CSMat,DMat,FMat,IMat,HMat,GMat,GIMat,GSMat,SMat,SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import scala.concurrent.future
//import scala.concurrent.ExecutionContext.Implicits.global
import scala.concurrent.ExecutionContextExecutor
import java.io._

/*
 * SFilesDatasource constructs SMat batches from data files stored on disk as IMat. 
 * The IMats are 3-column with column, row indices and integer values.
 * This format allows dynamic construction of the SMat with a specified bound on the max row index,
 * and with specified featurization (e.g. clipped to 1, linear, logarithmic etc.). 
 * fcounts is an IMat specifying the numbers of rows to use for each input block. 
 */

class SFileSourcev1(override val opts:SFileSource.Opts = new SFileSource.Options) extends FileSource(opts) {
  
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
    val threshold = opts.featThreshold
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
 //       println("here %d %d %d %d %d" format (k, mat.nrows, mat.ncols, lims.length, j))
        while (k < mat.nrows && mat.data(k) == irow && mat.data(k+mrows) < lims(j)) {
          if (xoff + k >= omat.ir.length) {
            throw new RuntimeException("SFileSource index out of range. Try increasing opts.eltsPerSample")
          }
          omat.ir(xoff + k) = mat.data(k+mrows) + yoff
          omat.data(xoff + k) = if (featType == 0) {
            1f
          } else if (featType == 1) {
            mat.data(k+2*mrows) 
          } else {
            if (mat.data(k+2*mrows).toDouble >= threshold.dv) 1f else 0f          
          }
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
      if (matq(i).asInstanceOf[AnyRef] != null) {
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
    while (todo > 0 && fileno < nend) {
    	var nrow = rowno
    	val filex = fileno % math.max(1, opts.lookahead)
    	if (opts.lookahead > 0) {
    	  while (ready(filex) < fileno) Thread.sleep(1); // `yield`
    	} else {
          fetch
        }
    	val spm = spmax(matqueue(filex)) + 1
//    	println("spm %d" format spm)
    	nrow = math.min(rowno + todo, spm)
    	val matq = matqueue(filex)
    	if (matq(0).asInstanceOf[AnyRef] != null) {
//    	  println("Here %d %d %d" format(rowno, nrow, todo))
    		omats(0) = sprowslice(matq, rowno, nrow, omats(0), opts.batchSize - todo)
    		if (rowno + todo >= spm) donextfile = true
    	} else {
    	  if (opts.throwMissing) {
    	    throw new RuntimeException("Missing file "+fileno)
    	  }
    	  donextfile = true
    	}
    	todo -= nrow - rowno
    	if (donextfile) {
    	  rowno = 0;
    	  fileno += 1;
    	  donextfile = false
    	} else {
    	  rowno = nrow;
    	}
    }
    if (todo > 0) {
      fillup(omats(0), todo)
    }
    omats
  }

}

/*
 * SFilesDatasource constructs SMat batches from data files stored on disk as IMat. 
 * The IMats are 3-column with column, row indices and integer values.
 * This format allows dynamic construction of the SMat with a specified bound on the max row index,
 * and with specified featurization (e.g. clipped to 1, linear, logarithmic etc.). 
 * fcounts is an IMat specifying the numbers of rows to use for each input block. 
 */

class SFileSource(override val opts:SFileSource.Opts = new SFileSource.Options) extends FileSource(opts) {
  
  var inptrs:IMat = null
  var offsets:IMat = null
  var fcounts:IMat = null
  
  override def init = {
    initbase
    fcounts = if (opts.fcounts == null) {
      val fc = izeros(opts.fnames.length,1)
      for (i <- 0 until opts.fnames.length) {
        val m = loadSMat(opts.fnames(0)(nstart))
        fc(i) = m.nrows
      }
      fc
    } else opts.fcounts
    var totsize = sum(fcounts).v
    if (opts.addConstFeat) totsize += 1
    omats = new Array[Mat](1)
    omats(0) = SMat(totsize, opts.batchSize, opts.batchSize * opts.eltsPerSample)
    inptrs = izeros(fcounts.length, 1)
    offsets = 0 on cumsum(fcounts)
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
  
  def spcolslice(inmat:Array[Mat], colno:Int, endcol:Int, omat0:Mat, done:Int):Mat = {
    val omat = omat0.asInstanceOf[SMat]
    val ioff = Mat.ioneBased
    var idone = done
    var innz = omat.nnz
    val lims = fcounts
    val nfiles = fcounts.length
    val addConstFeat = opts.addConstFeat
    val featType = opts.featType
    val threshold = opts.featThreshold
    var icol = colno;
    while (icol < endcol) {
      var j = 0;
      while (j < nfiles) {
        val mat = inmat(j).asInstanceOf[SMat];
        var k = mat.jc(icol) - ioff;
        var lastk = mat.jc(icol+1) - ioff;
        val xoff = innz - k;
 //       println("here %d %d %d %d %d" format (k, mat.nrows, mat.ncols, lims.length, j))
        while (k < lastk && mat.ir(k)-ioff < lims(j)) {
          if (xoff + k >= omat.ir.length) {
            throw new RuntimeException("SFileSource index out of range. Try increasing opts.eltsPerSample");
          }
          omat.ir(xoff + k) = mat.ir(k) + offsets(j);
          omat.data(xoff + k) = if (featType == 0) {
            1f;
          } else if (featType == 1) {
            mat.data(k) ;
          } else {
            if (mat.data(k).toDouble >= threshold.dv) 1f else 0f;       
          }
          k += 1;
        }
        innz = xoff + k
        j += 1
      }
      icol += 1
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
    var maxv = 0;
    for (i <- 0 until matq.length) {
      if (matq(i).asInstanceOf[AnyRef] != null) {
      	maxv = matq(i).ncols
      }
    }
    maxv - 1
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
    while (todo > 0 && fileno < nend) {
    	var nrow = rowno
    	val filex = fileno % math.max(1, opts.lookahead)
    	if (opts.lookahead > 0) {
    	  while (ready(filex) < fileno) Thread.sleep(1);// `yield`
    	} else {
    	  fetch
    	}    	
    	val spm = spmax(matqueue(filex)) + 1
//    	println("spm %d" format spm)
    	nrow = math.min(rowno + todo, spm)
    	val matq = matqueue(filex)
    	if (matq(0).asInstanceOf[AnyRef] != null) {
//    	  println("Here %d %d %d %d" format(rowno, nrow, todo, spm))
    		omats(0) = spcolslice(matq, rowno, nrow, omats(0), opts.batchSize - todo)
    		if (rowno + todo >= spm) donextfile = true
    	} else {
    	  if (opts.throwMissing) {
    	    throw new RuntimeException("Missing file "+fileno)
    	  }
    	  donextfile = true;
    	}
    	todo -= nrow - rowno
    	fprogress = nrow*1f / spm
    	if (donextfile) {
    	  rowno = 0;
    	  fileno += 1;
    	  fprogress = 0
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
  
  override def progress = {
    ((fileno-nstart)*1f + fprogress)/ totalSize
  }

}

object SFileSource {
  trait Opts extends FileSource.Opts {
  	var fcounts:IMat = null
  }
  
  class Options extends Opts {}

}

