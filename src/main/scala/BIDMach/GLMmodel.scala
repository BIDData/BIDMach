package BIDMach

import BIDMat.{Mat,BMat,CMat,DMat,FMat,IMat,HMat,GMat,GIMat,GSMat,SMat,SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._


class GLMmodel(opts:GLMmodel.Options) extends RegressionModel(opts) {
  
  var mylinks:IMat = null
  var targmap:Mat = null
  
  val linkArray = Array(LinearLink, LogisticLink)
  
  override def init(datasource:DataSource) = {
    super.init(datasource)
    mylinks = opts.links
    targmap = opts.targmap
  }
    
  def mupdate(in:Mat):FMat = {
    val prod = modelmats(0) * in
    val targ = targmap * prod.rowslice(0, targmap.ncols, null)
    val eta = prod.rowslice(targmap.ncols, prod.nrows - targmap.ncols, null)
    val pred = applylinks(eta)
    val update = (targ - pred) *^ in   
    if (opts.mask != null) update ~ update ∘ opts.mask
    updatemats(0) <-- update
    llfun(pred, targ)
  }
  
  def applylinks(eta:Mat):Mat = {
    eta match {
      case feta:FMat => {
    		var i = 0
    		val out = (feta + 1f)
    		while (i < feta.nrows) {
    			val fun = linkArray(mylinks(i)).invlinkfn
    		  var j = 0
    		  while (j < feta.ncols) {     			
    		  	out.data(i + j * out.nrows) = fun(feta.data(i + j * feta.nrows))
    		  	j += 1 
    		  }
    			i += 1
    		}
    		out
      }
    }
  }
  
  def llfun(pred:Mat, targ:Mat):FMat = {
    (targ, pred) match {
      case (ftarg:FMat, fpred:FMat) => {
    		var i = 0
    		val out = (ftarg + 1f)
    		while (i < ftarg.nrows) {
    			val fun = linkArray(mylinks(i)).likelihoodfn
    			var j = 0
    			while (j < ftarg.ncols) {
    				out.data(i + j * out.nrows) = fun(fpred.data(i + j * ftarg.nrows),  ftarg.data(i + j * ftarg.nrows))
    				j += 1
    			}
    			i += 1
    		}
    		out
      }
    }
  }

}


object LinearLink extends GLMlink {
  def link(in:Float) = {
    in
  }
  
  def invlink(in:Float) = {
    in
  }
  
  def dlink(in:Float) = {
    1.0f
  }
  
  def likelihood(pred:Float, targ:Float) = {
    val diff = targ - pred
    - diff * diff
  }
     
  override val linkfn = link _
  
  override val dlinkfn = dlink _
  
  override val invlinkfn = invlink _
  
  override val likelihoodfn = likelihood _
}

object LogisticLink extends GLMlink {
  def link(in:Float) = {
    math.log(in / (1.0f - in)).toFloat
  }
  
  def invlink(in:Float) = {
    val tmp = math.exp(in)
    (tmp / (1.0 + tmp)).toFloat
  }
  
  def dlink(in:Float) = {
    1 / (in * (1 - in))
  }
  
  def likelihood(pred:Float, targ:Float) = {
    math.log(targ * pred + (1.0f - targ) * (1.0f - pred)).toFloat
  }
  
  override val linkfn = link _
  
  override val dlinkfn = dlink _
  
  override val invlinkfn = invlink _
  
  override val likelihoodfn = likelihood _
}

object LinkEnum extends Enumeration {
  type LinkEnum = Value
  val Linear, Logistic = Value
}

abstract class GLMlink {
  val linkfn:(Float => Float)
  val dlinkfn:(Float => Float)
  val invlinkfn:(Float => Float)
  val likelihoodfn:((Float,Float) => Float)
}

object GLMmodel {
  class Options extends RegressionModel.Options {
    var links:IMat = null
    var targmap:Mat = null
    var mask:Mat = null
  }
  
  def learn = {
    
  }
}

