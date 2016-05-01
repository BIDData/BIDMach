package BIDMach.datasources
import BIDMat.{Mat,SBMat,CMat,CSMat,DMat,FMat,IMat,HMat,GMat,GIMat,GSMat,SMat,SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import BIDMat.MatIOtrait
import scala.concurrent.Future
import scala.concurrent.ExecutionContextExecutor
import java.io._

/** 
 *  Datasource designed to work with Iterators as provided by Spark. 
 *  We assume the iterator returns pairs from a Sequencefile of (StringWritable, MatIO)
 */

class IteratorSource(override val opts:IteratorSource.Opts = new IteratorSource.Options) extends DataSource(opts) { 
  var sizeMargin = 0f;
  var blockSize = 0;
  var samplesDone = 0;
  var nmats = 1;
  omats = null;
  var fprogress:Float = 0
  var inMats:Array[Mat] = null;
  var inFname:Array[String] = null;
  @transient var iter:Iterator[(AnyRef, MatIOtrait)] = null;
  var nblocks = -1;
  var iblock = 0;
  
  def reset = {
  	samplesDone = 0;
  	iblock = 0;
  }
  
  def init = {
  	samplesDone = 0;
  	iter = opts.iter;
  	blockSize = opts.batchSize;
  	iterNext;
  	nmats = inMats.length;
    inFname = new Array[String](nmats);
    omats = new Array[Mat](nmats);
    for (i <- 0 until nmats) {
      val mm = inMats(i);
      val (nr, nc) = if (opts.dorows) (blockSize, mm.ncols) else (mm.nrows, blockSize);
      omats(i) = mm match {
        case mf:FMat => FMat.newOrCheckFMat(nr, nc, null, GUID, i, ((nr*1L) << 32) + nc, "IteratorSource_FMat".##);
        case mi:IMat => IMat.newOrCheckIMat(nr, nc, null, GUID, i, ((nr*1L) << 32) + nc, "IteratorSource_IMat".##);
        case md:DMat => DMat.newOrCheckDMat(nr, nc, null, GUID, i, ((nr*1L) << 32) + nc, "IteratorSource_DMat".##);
        case ms:SMat => SMat.newOrCheckSMat(nr, nc, nc * opts.eltsPerSample, null, GUID, i, ((nr*1L) << 32) + nc, "IteratorSource_SMat".##);
      }
    } 
  }
  
  def next:Array[Mat] = {;
    var donextfile = false;
    var todo = blockSize;
    val featType = opts.featType;
    val threshold = opts.featThreshold;
    while (todo > 0) {
    	var samplesTodo = samplesDone;
    	var matqnr = 0
    	for (i <- 0 until nmats) {
    		val matq = inMats(i);
    		if (matq.asInstanceOf[AnyRef] != null) {
    		  matqnr = if (opts.dorows) matq.nrows else matq.ncols;
    		  samplesTodo = math.min(samplesDone + todo, matqnr);
    		  val off = Mat.oneBased
    		  if (opts.dorows) {
    		    val nc = omats(i).ncols;
    		    val nr = samplesTodo - samplesDone + blockSize - todo - off; 
    		    omats(i) = checkCaches(nr, nc, omats(i), GUID, i);                                         // otherwise, check for a cached copy
    		    omats(i) = matq.rowslice(samplesDone, samplesTodo, omats(i), blockSize - todo); 			  
    		  } else {
    		    val nr = omats(i).nrows;
    		    val nc = samplesTodo - samplesDone + blockSize - todo - off;
    		    omats(i) = checkCaches(nr, nc, omats(i), GUID, i); 
    		  	omats(i) = matq.colslice(samplesDone, samplesTodo, omats(i), blockSize - todo);
    		  }

    		  if (featType == 0) {
    		    min(1f, omats(i), omats(i));
    		  } else if (featType == 2) {
    		    omats(i) ~ omats(i) >= threshold;
    		  }
    		  if (matqnr == samplesTodo) donextfile = true;
    		} else {
    		  donextfile = true;
    		}
    	}
    	todo -= samplesTodo - samplesDone;
    	if (donextfile) {
    	  samplesDone = 0;
    	  if (iterHasNext) {
    	    iterNext();
    	  }
    	  donextfile = false;
    	} else {
    	  samplesDone = samplesTodo;
    	}
    	fprogress = samplesDone*1f / matqnr;
    }
    omats;
  }
  
  def progress:Float = {
    if (nblocks > 0) {
    	(fprogress + iblock-1)/nblocks;
    } else 0f
  }
  
  def hasNext:Boolean = {
    val matq = inMats(0);
    val matqnr = if (opts.dorows) matq.nrows else matq.ncols;
    val ihn = iter.hasNext;
    if (! ihn && iblock > 0) {
      nblocks = iblock;
    }
    (ihn || (matqnr - samplesDone) == 0);
  }
  
  def iterHasNext:Boolean = {
  	iblock += 1;
    iter.hasNext;
  }
  
  def iterNext() = {
    inMats = iter.next._2.get
  }
  
  def lazyTranspose(a:Mat) = {
    a match {
      case af:FMat => FMat(a.ncols, a.nrows, af.data)
      case ad:DMat => DMat(a.ncols, a.nrows, ad.data)
      case ai:IMat => IMat(a.ncols, a.nrows, ai.data)
      case _ => throw new RuntimeException("laztTranspose cant deal with "+a.getClass.getName)
    }
  }
  
  def checkCaches(nr:Int, nc:Int, out:Mat, GUID:Long, i:Int):Mat = {
    if (nr == out.nrows && nc == out.ncols) {
      out 
    } else {
    	out match {
    	case a:FMat => FMat.newOrCheckFMat(nr, nc, null, GUID, i, ((nr*1L) << 32) + nc, "IteratorSource_FMat".##);
    	case a:IMat => IMat.newOrCheckIMat(nr, nc, null, GUID, i, ((nr*1L) << 32) + nc, "IteratorSource_IMat".##);
    	case a:DMat => DMat.newOrCheckDMat(nr, nc, null, GUID, i, ((nr*1L) << 32) + nc, "IteratorSource_DMat".##);
    	case a:SMat => SMat.newOrCheckSMat(nr, nc, a.nnz, null, GUID, i, ((nr*1L) << 32) + nc, "IteratorSource_SMat".##);
    	}
    }
  }
  

  override def close = {
    inMats = null;
//    stop = true
  }
}


object IteratorSource {
  
  def apply(opts:IteratorSource.Opts):IteratorSource = {
    new IteratorSource(opts);
  } 
 
  trait Opts extends DataSource.Opts {
    var nmats = 1;
    var dorows:Boolean = false
    @transient var iter:Iterator[Tuple2[AnyRef, MatIOtrait]] = null;
    var eltsPerSample = 10;
    var throwMissing:Boolean = false; 
  }
  
  class Options extends Opts {}
}
