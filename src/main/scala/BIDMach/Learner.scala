package BIDMach
import BIDMat.{Mat,BMat,CMat,DMat,FMat,IMat,HMat,GMat,GIMat,GSMat,SMat,SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import BIDMat.Plotting._
import Learner._
import scala.collection.immutable.List

case class Learner(datamat:Mat, targetmat:Mat, model:Model, regularizer:Regularizer, updater:Updater, val opts:Learner.Options = new Learner.Options) {

  val n = datamat.ncols
  val options = opts
  val nw = options.memwindow/options.blocksize
  val nww = options.convwindow/options.blocksize
  var tscores:List[Float] = List()
  var tsteps:List[Float] = List()

  def run() = {
  	var done:Boolean = false
  	var ipass = 0
  	var llest:Double = 0
  	var llder:Double = 0
  	var llold:Double = 0
  	var tsecs:Double = 1
  	var nsteps:Long = 0
  	tic
  	while (ipass < options.npasses && ! done) {
  		var i = 0
  		while (i < n && ! done) {
  			var iend = math.min(n, i+options.blocksize)
  			nsteps += iend - i
  			var dslice = datamat(?, i->iend)
  			var tslice = targetmat(?, i->iend)
  			val tll = model.gradfun(dslice, tslice)
  			regularizer.compute(iend-i)
  			updater.update(iend-i)

  			llest = (1/nw)*(tll + (nw-1)*llest)
  			llder = (1/nww)*(tll-llold + (nww-1)*llder)
  			llold = tll
  			i += options.blocksize
  			if (llder > 0 && llder < options.convslope) {
  				done = true
  			}
  			if (toc >= tsecs || done || (ipass == options.npasses-1 && i >= n)) {
  				println("pass=%d, i=%d t=%3.1f secs, ll=%5.4f, slope=%5.4g" format (ipass, i, tsecs, llest, llder))
  				tscores = tscores :+ -llest.asInstanceOf[Float]
  				tsteps = tsteps :+ nsteps.asInstanceOf[Float]
  			  tsecs += options.secprint
  			}
  		}
  		ipass += 1
  	}
  val timeplot = plot(1->(tscores.size+1), row(tscores.toArray))
  val stepplot = plot(row(tsteps.toArray), row(tscores.toArray))
  timeplot.setTitle("Neg log likelihood vs time in seconds")
  stepplot.setTitle("Neg log likelihood vs number of samples")
  }
}


object Learner {
	class Options {
		var blocksize:Int = 8000
		var npasses:Int = 100
	  var memwindow:Double = 1000000
		var convwindow:Double = 1000000	
		var convslope:Double = -1e-6
		var secprint:Double = 1
		var eps:Float = 1e-8f
  }
	
  def checkSize(a:Mat, nr:Int, nc:Int, b:Mat):Mat = {
    if (a.asInstanceOf[AnyRef] != null && a.nrows == nr && a.ncols == nc) {
      a
    } else {
      b.zeros(nr, nc)
    }
  }
  
  def checkSize(a:Mat, b:Mat):Mat = checkSize(a, b.nrows, b.ncols, b)
  
  def fsqrt(v:Float):Float = math.sqrt(v).asInstanceOf[Float]
  
  def mapfun2x2(fn:(Float, Float)=>(Float, Float), in0:FMat, in1:FMat, out0:FMat, out1:FMat) = {
    if (in0.nrows != in1.nrows || in0.nrows != out0.nrows || in0.nrows != out1.nrows ||
        in0.ncols != in1.ncols || in0.ncols != out0.ncols || in0.ncols != out1.ncols) {
      throw new RuntimeException("dimensions mismatch")
    }
    var i = 0
    while (i < in0.length) {
      val (v1, v2) = fn(in0.data(i), in1.data(i))
      out0.data(i) = v1
      out1.data(i) = v2
      i += 1
    }
  }
  def mapfun2x1(fn:(Float, Float)=>Float, in0:FMat, in1:FMat, out0:FMat) = {
    if (in0.nrows != in1.nrows || in0.nrows != out0.nrows ||
        in0.ncols != in1.ncols || in0.ncols != out0.ncols) {
      throw new RuntimeException("dimensions mismatch")
    }
    var i = 0
    while (i < in0.length) {
      out0.data(i) = fn(in0.data(i), in1.data(i))
      i += 1
    }
  }
}
