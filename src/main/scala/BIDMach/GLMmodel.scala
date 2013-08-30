package BIDMach

import BIDMat.{Mat,BMat,CMat,DMat,FMat,IMat,HMat,GMat,GIMat,GSMat,SMat,SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._


class GLMmodel(opts:GLMmodel.Options) extends RegressionModel(opts) {
  
  var mylinks:Mat = null
  
  val linkArray = Array(LinearLink, LogisticLink)
  
  override def init(datasource:DataSource) = {
    super.init(datasource)
    mylinks = if (useGPU) GIMat(opts.links) else opts.links
    modelmats(0) ~ modelmats(0) ∘ mask
  }
    
  def mupdate(in:Mat):FMat = {
    val eta = modelmats(0) * in
//    println("model %f" format (mean(mean(modelmats(0)))).dv)
    val targ = targmap * (targets * in)
    val pred = applylinks(eta, mylinks)
//    println("pred %f" format (mean(mean(pred))).dv)
    val update = (targ - pred) *^ in
    if (mask != null) update ~ update ∘ mask
    updatemats(0) <-- update
    val lls = llfun(pred, targ, mylinks)
    mean(lls, 2)
  }
  
  def applylinks(eta:Mat, links:Mat):Mat = {
    (eta, links) match {
      case (feta:FMat, ilinks:IMat) => {
        Mat.nflops += 10L * feta.length
    		var i = 0
    		val out = (feta + 1f)
    		while (i < feta.ncols) {
    		  var j = 0
    		  while (j < feta.nrows) { 
    		  	val fun = linkArray(ilinks(j)).invlinkfn
    		  	out.data(j + i * out.nrows) = fun(feta.data(j + i * feta.nrows))
    		  	j += 1 
    		  }
    			i += 1
    		}
    		out
      }
    }
  }
  
  def llfun(pred:Mat, targ:Mat, links:Mat):FMat = {
    (pred, targ, links) match {
      case (fpred:FMat, ftarg:FMat, ilinks:IMat) => {
      	Mat.nflops += 10L * ftarg.length
    		var i = 0
    		val out = (ftarg + 1f)
    		while (i < ftarg.ncols) {
    			var j = 0
    			while (j < ftarg.nrows) {
    				val fun = linkArray(ilinks(j)).likelihoodfn
    				out.data(j + i * out.nrows) = fun(fpred.data(j + i * ftarg.nrows),  ftarg.data(j + i * ftarg.nrows))
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
  }
  
}

