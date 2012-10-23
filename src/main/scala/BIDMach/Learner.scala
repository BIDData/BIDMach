package BIDMach
import BIDMat.{Mat,BMat,CMat,DMat,FMat,IMat,HMat,GMat,GIMat,GSMat,SMat,SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import Learner._

trait Model {
  var options:Learner.Options = null
  def initmodel(learner:Learner, data:Mat, target:Mat):Mat
  def gradfun(data:Mat, target:Mat, model:Mat, diff:Mat):Double
}

case class Learner(datamat:Mat, targetmat:Mat, model:Model, optimizer:Optimizer) {

  var options:Options = model.options
  var modelmat:Mat = null	
	model.initmodel(this, datamat, targetmat)
  val (nrows, ncols) = size(modelmat)
  val diff:Mat = modelmat.zeros(nrows, ncols)
  val n = datamat.ncols
  val nw = options.memwindow/options.blocksize
  val nww = options.convwindow/options.blocksize
  optimizer.init(this, modelmat)
  
  def run() = {
  	var done:Boolean = false
  	var ipass = 0
  	var llest:Double = 0
  	var llder:Double = 0
  	var llold:Double = 0
  	var tsecs:Double = 1
  	tic
  	while (ipass < options.npasses && ! done) {
  		var i = 0
  		while (i < n && ! done) {
  			var iend = math.min(n, i+options.blocksize)
  			var dslice = datamat(?, i->iend)
  			var tslice = targetmat(?, i->iend)
  			val tll = model.gradfun(dslice, tslice, modelmat, diff)
  			optimizer.update(modelmat, diff, options.blocksize)

  			llest = (1/nw)*(tll + (nw-1)*llest)
  			llder = (1/nww)*(tll-llold + (nww-1)*llder)
  			llold = tll
  			i += options.blocksize
  			if (llder > 0 && llder < options.convslope) {
  				done = true
  			}
  			if (toc >= tsecs || done || (ipass == options.npasses-1 && i >= n)) {
  				println("pass=%d, i=%d t=%3.1f secs, ll=%5.4f, slope=%5.4g, norm=%5.2f" format 
  				    (ipass, i, tsecs, llest, llder, norm(modelmat)))
  				tsecs += options.secprint
  			}
  		}
  		ipass += 1
  	}
  }
}


object Learner {
	class Options {
		var blocksize:Int = 10000
		var npasses:Int = 100
		var alpha:Float = 500f
		var memwindow:Double = 1000000
		var convwindow:Double = 1000000	
		var gradwindow:Float = 10000000
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
