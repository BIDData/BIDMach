package BIDMach

import BIDMat.{Mat,BMat,CMat,DMat,FMat,IMat,HMat,GMat,GIMat,GSMat,SMat,SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import edu.berkeley.bid.CUMAT


class GLMmodel(opts:GLMmodel.Options) extends RegressionModel(opts) {
  
  var mylinks:Mat = null
  
  val linkArray = Array[GLMlink](LinearLink, LogisticLink)
  
  var totflops = 0L
  
  override def init(datasource:DataSource) = {
    super.init(datasource)
    mylinks = if (useGPU) GIMat(opts.links) else opts.links
    modelmats(0) ~ modelmats(0) ∘ mask
    totflops = 0L
    for (i <- 0 until opts.links.length) {
      totflops += linkArray(opts.links(i)).fnflops
    }
  }
    
  def mupdate(in:Mat):FMat = {
//    println("model %f" format (mean(mean(modelmats(0)))).dv)
    val targs = targets * in
    min(targs, 1f, targs)
    val alltargs = targmap * targs
    val eta = modelmats(0) * in
    applymeans(eta, mylinks, eta)
//    println("pred %f" format (mean(mean(pred))).dv)
//    println("%s %s %s %s %s" format (modelmats(0).mytype, updatemats(0).mytype, alltargs.mytype, pred.mytype, in.mytype))
    val lls = llfun(eta, alltargs, mylinks)
    alltargs ~ alltargs - eta
    updatemats(0) ~ alltargs *^ in
    lls
  }
  
  def applymeans(eta:Mat, links:Mat, out:Mat):Mat = {
    (eta, links, out) match {
      case (feta:FMat, ilinks:IMat, fout:FMat) => {
        Mat.nflops += totflops * feta.ncols
    		var i = 0
    		val out = (feta + 3f)
    		while (i < feta.ncols) {
    		  var j = 0
    		  while (j < feta.nrows) { 
    		  	val fun = linkArray(ilinks(j)).invlinkfn
    		  	fout.data(j + i * out.nrows) = fun(feta.data(j + i * feta.nrows))
    		  	j += 1 
    		  }
    			i += 1
    		}
    		out
      }
      case (geta:GMat, gilinks:GIMat, gout:GMat) => {
      	Mat.nflops += totflops * geta.ncols
      	CUMAT.applymeans(geta.data, gilinks.data, gout.data, geta.nrows, geta.ncols)
      	out
      }
    }
  }
  
  def llfun(pred:Mat, targ:Mat, links:Mat):FMat = {
    (pred, targ, links) match {
      case (fpred:FMat, ftarg:FMat, ilinks:IMat) => {
      	Mat.nflops += 10L * ftarg.length
    		var i = 0
    		val out = (ftarg + 5f)
    		while (i < ftarg.ncols) {
    			var j = 0
    			while (j < ftarg.nrows) {
    				val fun = linkArray(ilinks(j)).likelihoodfn
    				out.data(j + i * out.nrows) = fun(fpred.data(j + i * ftarg.nrows),  ftarg.data(j + i * ftarg.nrows))
    				j += 1
    			}
    			i += 1
    		}
    		mean(out,2)
      }
      case (gpred:GMat, gtarg:GMat, gilinks:GIMat) => {
      	Mat.nflops += totflops * gpred.ncols
      	val out = (gpred + 3f)
      	CUMAT.applylls(gpred.data, gtarg.data, gilinks.data, out.data, gpred.nrows, gpred.ncols)
      	FMat(mean(out,2))
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
  
  val fnflops = 2
}

object LogisticLink extends GLMlink {
  def link(in:Float) = {
    math.log(in / (1.0f - in)).toFloat
  }
  
  def invlink(in:Float) = {
    if (in > 0) {
    	val tmp = math.exp(-in)
    	(1.0 / (1.0 + tmp)).toFloat    
    } else {
    	val tmp = math.exp(in)
    	(tmp / (1.0 + tmp)).toFloat
    }
  }
  
  def dlink(in:Float) = {
    1 / (in * (1 - in))
  }
  
  def likelihood(pred:Float, targ:Float) = {
    math.log(targ * pred + (1.0f - targ) * (1.0f - pred) + 1e-20).toFloat
  }
  
  override val linkfn = link _
  
  override val dlinkfn = dlink _
  
  override val invlinkfn = invlink _
  
  override val likelihoodfn = likelihood _
  
  val fnflops = 20
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
  val fnflops:Int
}

object GLMmodel {
  class Options extends RegressionModel.Options {
    var links:IMat = null
  }
  
  def mkGLMmodel(fopts:Model.Options) = {
  	new GLMmodel(fopts.asInstanceOf[GLMmodel.Options])
  }
  
  def mkUpdater(nopts:Updater.Options) = {
  	new ADAGradUpdater(nopts.asInstanceOf[ADAGradUpdater.Options])
  }
  
  def learnFParx(
    nstart:Int=FilesDataSource.encodeDate(2012,3,1,0),
		nend:Int=FilesDataSource.encodeDate(2012,12,1,0)
		) = {	
  	new LearnFParModelx(
  	    SFilesDataSource.twitterNgramBlend(nstart, nend, 1, 0),
  	    new GLMmodel.Options, mkGLMmodel _, 
  	    new ADAGradUpdater.Options, mkUpdater _)
  }
}

