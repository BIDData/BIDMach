package BIDMach.datasources
import BIDMat.{Mat,SBMat,CMat,CSMat,DMat,FMat,IMat,HMat,GMat,GIMat,GSMat,SMat,SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import BIDMat.MatIOtrait
import scala.concurrent.Future
import scala.concurrent.ExecutionContextExecutor
import java.io._

class ArraySource(override val opts:ArraySource.Opts = new ArraySource.Options) extends IteratorSource(opts) {
  @transient var dataArray:Array[_ <: AnyRef] = null

  override def init = {
    dataArray = opts.dataArray
    super.init
  }

  override def iterHasNext:Boolean = {
    iblock += 1
    iblock < dataArray.length
  }

  override def hasNext:Boolean = {
    val matq = inMats(0)
    val matqnr = if (opts.dorows) matq.nrows else matq.ncols
    val ihn = iblock < dataArray.length
    if (! ihn && iblock > 0) {
      nblocks = iblock
    }
    (ihn || (matqnr - samplesDone) == 0);
  }

  override def iterNext() = {
    val marr = dataArray(iblock)
    marr match {
      case (key:AnyRef,v:MatIOtrait) => {inMats = v.get}
      case m:Mat => {
        if (inMats == null) inMats = Array[Mat](1);
        inMats(0) = m;
      }
      case ma:Array[Mat] => inMats = ma;
    }
  }

  override def close = {
    iblock = 0
  }
}

object ArraySource {
  def apply(opts:ArraySource.Opts):ArraySource = {
    new ArraySource(opts);
  }

  trait Opts extends IteratorSource.Opts {
    @transient var dataArray:Array[_ <: AnyRef] = null
  }

  class Options extends Opts {}
}
