package BIDMach.datasources
import BIDMat.{Mat,SBMat,CMat,CSMat,DMat,FMat,IMat,HMat,GMat,GIMat,GSMat,SMat,SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import scala.concurrent.Future
import scala.concurrent.ExecutionContextExecutor
import java.io._

class FilesDS(override val opts:FilesDS.Opts = new FilesDS.Options)(implicit val ec:ExecutionContextExecutor) extends DataSource(opts) { 
  var sizeMargin = 0f
  var blockSize = 0 
  @volatile var fileno = 0
  var rowno = 0
  var nstart = 0
  var nend = 0
  var fnames:List[(Int)=>String] = null
  omats = null
  var matqueue:Array[Array[Mat]] = null
  var ready:IMat = null
  var stop:Boolean = false
  var permfn:(Int)=>Int = null
  var totalSize = 0
  var fprogress:Float = 0
  var lastMat:Array[Mat] = null;
  var lastFname:Array[String] = null;
  
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
  
  def genperm(nstart:Int, nend:Int) = {
    val (dmy, ii) = sort2(rand(nend - nstart - 1,1));
    (n:Int) => {
      if (n >= nend - 1) {
        n
      } else {
        nstart + ii(n - nstart, 0);
      }
    }
  }
  
  def initbase = {
    ready = -iones(math.max(opts.lookahead,1), 1)                              // Numbers of files currently loaded in queue
    reset    
    rowno = 0;
    fileno = nstart;                                                           // Number of the current output file      
    matqueue = new Array[Array[Mat]](math.max(1,opts.lookahead))               // Queue of matrices for each output matrix
    for (i <- 0 until math.max(1,opts.lookahead)) {
      matqueue(i) = new Array[Mat](fnames.size);
    }
    if (opts.putBack < 0) {
    	for (i <- 0 until opts.lookahead) {
    		Future {
    			prefetch(nstart + i);
    		}
    	}
    }
  }
  
  def reset = {
    nstart = opts.nstart
    nend = opts.nend
    fnames = opts.fnames
    blockSize = opts.batchSize
    if (nend == 0) {
      while (fileExists(fnames(0)(nend))) {nend += 1}
    }
    while (!fileExists(fnames(0)(nstart)) && nstart < nend) {nstart += 1}
    if (nstart == nend) {
      throw new RuntimeException("Couldnt find any files");
    }
    if (opts.order == 0) {
      permfn = (a:Int) => a
    } else if (opts.order == 1) {
      permfn = genperm(nstart, nend)
    } else {
      permfn = (n:Int) => {                                                    // Stripe reads across disks (different days)
        val (yy, mm, dd, hh) = FilesDS.decodeDate(n)
        val hhdd = hh + 24 * (dd - 1)
        FilesDS.encodeDate(yy, mm, hhdd % 31 + 1, hhdd / 31)
      } 
    }
    rowno = 0;
    fileno = nstart;
    for (i <- 0 until math.max(1,opts.lookahead)) {
      val ifile = nstart + i
      val ifilex = ifile % math.max(opts.lookahead, 1)
      ready(ifilex) = ifile - math.max(1, opts.lookahead)
    } 
    totalSize = nend - nstart;
    lastMat = new Array[Mat](fnames.size);
    lastFname = new Array[String](fnames.size);
    for (i <- 0 until lastMat.length) {lastMat(i) = null;}
    for (i <- 0 until lastFname.length) {lastFname(i) = null;}
  }
  
  def init = {
    initbase
    omats = new Array[Mat](fnames.size)
    for (i <- 0 until fnames.size) {
      var mm = HMat.loadMat(fnames(i)(nstart));
      val (nr, nc) = if (opts.dorows) (blockSize, mm.ncols) else (mm.nrows, blockSize);
      omats(i) = mm match {
        case mf:FMat => FMat.newOrCheckFMat(nr, nc, null, GUID, i, ((nr*1L) << 32) + nc, "FilesDS_FMat".##);
        case mi:IMat => IMat.newOrCheckIMat(nr, nc, null, GUID, i, ((nr*1L) << 32) + nc, "FilesDS_IMat".##);
        case md:DMat => DMat.newOrCheckDMat(nr, nc, null, GUID, i, ((nr*1L) << 32) + nc, "FilesDS_DMat".##);
        case ms:SMat => SMat.newOrCheckSMat(nr, nc, nc * opts.eltsPerSample, null, GUID, i, ((nr*1L) << 32) + nc, "FilesDS_SMat".##);
      }
    } 
  }
  
  def progress = {
    ((fileno-nstart)*1f + fprogress)/ totalSize
  }
  
  def nmats = omats.length
  
  def next:Array[Mat] = {
    var donextfile = false;
    var todo = blockSize;
    val featType = opts.featType;
    val threshold = opts.featThreshold;
    while (todo > 0 && fileno < nend) {
    	var nrow = rowno;
    	val filex = fileno % math.max(1, opts.lookahead);
//    	        println("todo %d, fileno %d, filex %d, rowno %d" format (todo, fileno, filex, rowno))
    	if (opts.putBack < 0 && opts.lookahead > 0) {
    	  while (ready(filex) < fileno) Thread.`yield`
    	} else {
    	  fetch
    	}
    	var matqnr = 0
    	for (i <- 0 until fnames.size) {
    		val matq = matqueue(filex)(i);
    		if (matq.asInstanceOf[AnyRef] != null) {
    		  matqnr = if (opts.dorows) matq.nrows else matq.ncols;
    		  nrow = math.min(rowno + todo, matqnr);
    		  val off = Mat.oneBased
    		  if (opts.dorows) {
    		    val nc = omats(i).ncols;
    		    val nr = nrow - rowno + blockSize - todo - off; 
    		    omats(i) = checkCaches(nr, nc, omats(i), GUID, i);                                         // otherwise, check for a cached copy
    		    omats(i) = matq.rowslice(rowno, nrow, omats(i), blockSize - todo); 			  
    		  } else {
    		    val nr = omats(i).nrows;
    		    val nc = nrow - rowno + blockSize - todo - off;
    		    omats(i) = checkCaches(nr, nc, omats(i), GUID, i); 
    		  	omats(i) = matq.colslice(rowno, nrow, omats(i), blockSize - todo);
    		  }

    		  if (featType == 0) {
    		    min(1f, omats(i), omats(i));
    		  } else if (featType == 2) {
    		    omats(i) ~ omats(i) >= threshold;
    		  }
    		  if (matqnr == nrow) donextfile = true;
    		} else {
    		  if (opts.throwMissing) {
    		    throw new RuntimeException("Missing file "+fileno);
    		  }
    		  donextfile = true;
    		}
    	}
    	todo -= nrow - rowno;
    	if (donextfile) {
    	  rowno = 0;
    	  fileno += 1;
    	  donextfile = false;
    	} else {
    	  rowno = nrow;
    	}
    	fprogress = rowno*1f / matqnr;
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
      while (ready(ifilex) >= fileno && !stop) Thread.`yield`
      if (!stop) {
        val inew = ready(ifilex) + opts.lookahead;
        val pnew = permfn(inew);
        val fexists = fileExists(fnames(0)(pnew)) && (rand(1,1).v <= opts.sampleFiles);
        for (i <- 0 until fnames.size) {
          matqueue(ifilex)(i) = if (fexists) {
            HMat.loadMat(fnames(i)(pnew), matqueue(ifilex)(i));	
          } else {
            if (opts.throwMissing && inew < nend) {
              throw new RuntimeException("Missing file "+fnames(i)(pnew));
            }
            null;  	
          }
          //  			println("%d" format inew)
        }
        ready(ifilex) = inew;
      }
  	}
  }
  
  def checkCaches(nr:Int, nc:Int, out:Mat, GUID:Long, i:Int):Mat = {
    if (nr == out.nrows && nc == out.ncols) {
      out 
    } else {
    	out match {
    	case a:FMat => FMat.newOrCheckFMat(nr, nc, null, GUID, i, ((nr*1L) << 32) + nc, "FilesDS_FMat".##);
    	case a:IMat => IMat.newOrCheckIMat(nr, nc, null, GUID, i, ((nr*1L) << 32) + nc, "FilesDS_IMat".##);
    	case a:DMat => DMat.newOrCheckDMat(nr, nc, null, GUID, i, ((nr*1L) << 32) + nc, "FilesDS_DMat".##);
    	case a:SMat => SMat.newOrCheckSMat(nr, nc, a.nnz, null, GUID, i, ((nr*1L) << 32) + nc, "FilesDS_SMat".##);
    	}
    }
  }
  
  def fetch = {
    if (ready(0) < fileno) {
      val pnew = permfn(fileno);
      val fexists = fileExists(fnames(0)(pnew)) && (rand(1,1).v <= opts.sampleFiles);
      for (i <- 0 until fnames.size) {
        if (fexists && lastMat(i).asInstanceOf[AnyRef] != null) {
          HMat.saveMat(lastFname(i), lastMat(i));
        }
        matqueue(0)(i) = if (fexists) {
          val tmp = HMat.loadMat(fnames(i)(pnew), matqueue(0)(i));
          lastFname(i) = fnames(i)(pnew);
          lastMat(i) = tmp;
          tmp;
        } else {
          if ((opts.sampleFiles >= 1.0f) && opts.throwMissing) {
            throw new RuntimeException("Missing file "+fnames(i)(pnew));
          }
          null;              
        }
      }
      ready(0) = fileno;
    }
  }

  
  def hasNext:Boolean = {
    (fileno < nend)
  }

  override def close = {
//    stop = true
  }
}


object FilesDS {
  
  def apply(opts:FilesDS.Opts, nthreads:Int):FilesDS = {
    implicit val ec = threadPool(nthreads);
    new FilesDS(opts);
  }
  
  def apply(opts:FilesDS.Opts):FilesDS = apply(opts, 4);
  
  def apply(fname:String, opts:FilesDS.Opts, nthreads:Int):FilesDS = {
    opts.fnames = List(simpleEnum(fname, 1, 0));
    implicit val ec = threadPool(nthreads);
    new FilesDS(opts);
  }
  
  def apply(fname:String, opts:FilesDS.Opts):FilesDS = apply(fname, opts, 4);
  
  def apply(fname:String):FilesDS = apply(fname, new FilesDS.Options, 4);
  
  def apply(fn1:String, fn2:String, opts:FilesDS.Opts, nthreads:Int) = {
    opts.fnames = List(simpleEnum(fn1, 1, 0), simpleEnum(fn2, 1, 0));
    implicit val ec = threadPool(nthreads);
    new FilesDS(opts);
  }
  
  def apply(fn1:String, fn2:String, opts:FilesDS.Opts):FilesDS = apply(fn1, fn2, opts, 4);
  
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
  
  def simpleEnum(fname:String, m:Int, i:Int):(Int)=>String = {
    (n0:Int) => { 
      val n = n0 * m + i
      (fname format n)
    }    
  }
 
  trait Opts extends DataSource.Opts {
  	val localDir:String = ""
  	var fnames:List[(Int)=>String] = null
  	var lookahead = 2
  	var sampleFiles = 1.0f
    var nstart:Int = 0
    var nend:Int = 0
    var dorows:Boolean = false
    var order:Int = 0                          // 0 = sequential order, 1 = random
    var eltsPerSample = 10;
    var throwMissing:Boolean = false
  }
  
  class Options extends Opts {}
}
