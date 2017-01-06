package BIDMach.datasources
import BIDMat.{Mat,SBMat,CMat,CSMat,DMat,FMat,Filter,FND,IMat,HMat,GMat,GIMat,GSMat,GND,ND,SMat,SDMat}
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
  var inMats:Array[ND] = null;
  var inFname:Array[String] = null;
  @transient var iter:Iterator[AnyRef] = null;
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
    omats = new Array[ND](nmats);
    for (i <- 0 until nmats) {
      val mm = inMats(i);
      val (nr, nc) = (mm.nrows, blockSize);
      omats(i) = mm match {
        case mf:FMat => FMat.newOrCheckFMat(nr, nc, null, GUID, i, ((nr*1L) << 32) + nc, "IteratorSource_FMat".##);
        case mi:IMat => IMat.newOrCheckIMat(nr, nc, null, GUID, i, ((nr*1L) << 32) + nc, "IteratorSource_IMat".##);
        case md:DMat => DMat.newOrCheckDMat(nr, nc, null, GUID, i, ((nr*1L) << 32) + nc, "IteratorSource_DMat".##);
        case ms:SMat => SMat.newOrCheckSMat(nr, nc, nc * opts.eltsPerSample, null, GUID, i, ((nr*1L) << 32) + nc, "IteratorSource_SMat".##);
        case a:FND => {
        	val newdims = mm.dims.copy;
        	newdims(newdims.length-1) = nc;
        	val hmm = Filter.hashIMat(newdims);
        	FND.newOrCheckFND(newdims, null, GUID, i, hmm, "IteratorSource_FND".##);
        }
      }
    }
  }

  def next:Array[ND] = {;
    var donextfile = false;
    var todo = blockSize;
    val featType = opts.featType;
    val threshold = opts.featThreshold;
    while (todo > 0) {
      var samplesTodo = samplesDone;
      var matqnc = 0
      for (i <- 0 until nmats) {
        val matq = inMats(i);
        if (matq.asInstanceOf[AnyRef] != null) {
          matqnc = matq.ncols;
          samplesTodo = math.min(samplesDone + todo, matqnc);
          val off = Mat.oneBased;
          val nr = omats(i).nrows;
          val nc = samplesTodo - samplesDone + blockSize - todo - off;
          omats(i) = checkCaches(nc, omats(i), GUID, i);
          omats(i) = matq.colslice(samplesDone, samplesTodo, omats(i), blockSize - todo);

          if (featType == 0) {
            min(omats(i), 1f, omats(i));
          } else if (featType == 2) {
            omats(i) ~ omats(i) >= threshold;
          }
          if (matqnc == samplesTodo) donextfile = true;
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
      fprogress = samplesDone*1f / matqnc;
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
    val matqnc = matq.ncols;
    val ihn = iter.hasNext;
    if (! ihn && iblock > 0) {
      nblocks = iblock;
    }
    (ihn || (matqnc - samplesDone) == 0);
  }

  def iterHasNext:Boolean = {
    iblock += 1;
    iter.hasNext;
  }

  def iterNext() = {
    val marr = iter.next;
    marr match {
      case (key:AnyRef,v:MatIOtrait) => {inMats = v.get}
      case m:ND => {
        if (inMats == null) inMats = Array[ND](1);
        inMats(0) = m;
      }
      case ma:Array[ND] => inMats = ma;
    }
  }

 def checkCaches(nc:Int, out:ND, GUID:Long, i:Int):ND = {
    val dims = out.dims;
    val nr = out.nrows;
    if (nc == out.ncols) {
      out
    } else {
      out match {
        case a:FMat => FMat.newOrCheckFMat(nr, nc, null, GUID, i, ((nr*1L) << 32) + nc, "IteratorSource_FMat".##);
        case a:IMat => IMat.newOrCheckIMat(nr, nc, null, GUID, i, ((nr*1L) << 32) + nc, "IteratorSource_IMat".##);
        case a:DMat => DMat.newOrCheckDMat(nr, nc, null, GUID, i, ((nr*1L) << 32) + nc, "IteratorSource_DMat".##);
        case a:SMat => SMat.newOrCheckSMat(nr, nc, a.nnz, null, GUID, i, ((nr*1L) << 32) + nc, "IteratorSource_SMat".##);
        case a:FND => {
          val newdims = dims.copy;
          newdims(dims.length-1) = nc;
        	val hmm = Filter.hashIMat(newdims);
          FND.newOrCheckFND(newdims, null, GUID, i, hmm, "IteratorSource_FND".##);
        }
      }
    }
  }


  override def close = {
    inMats = null;
    omats = null
    opts.iter = null
    iter = null
    //    stop = true
  }
}


object IteratorSource {

  def apply(opts:IteratorSource.Opts):IteratorSource = {
    new IteratorSource(opts);
  }

  trait Opts extends DataSource.Opts {
    var nmats = 1;
    @transient var iter:Iterator[Tuple2[AnyRef, MatIOtrait]] = null;
    var eltsPerSample = 10;
    var throwMissing:Boolean = false;
  }

  class Options extends Opts {}
}
