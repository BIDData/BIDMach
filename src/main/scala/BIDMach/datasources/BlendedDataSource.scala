package BIDMach.datasources
import BIDMat.{Mat,BMat,CMat,CSMat,DMat,FMat,IMat,HMat,GMat,GIMat,GSMat,SMat,SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import scala.actors._
import java.io._

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


object BlendedDataSource {
  trait Opts extends DataSource.Opts {
  	var bBlock = 1000
  }
  
  class Options extends Opts {}
}

