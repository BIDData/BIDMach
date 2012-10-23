package BIDMach
import BIDMat.{Mat,BMat,CMat,DMat,FMat,IMat,HMat,GMat,GIMat,GSMat,SMat,SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import Learner._

class LinearRegModel extends RegressionModel {
  class Options extends Learner.Options {}
  options = new Options
  
  options.blocksize = 8000
  options.npasses = 10
  options.alpha = 200f
  options.convslope = -1e-6
  
  var diff:Mat = null
  var tmp0:Mat = null
       
  override def regfn(targ:Mat, pred:Mat, lls:Mat, gradw:Mat):Unit =
    linearMap1(targ, pred, lls, gradw)

  def linearMap1(targ:Mat, pred:Mat, lls:Mat, gradw:Mat):Unit= {
  	diff = checkSize(diff, targ)
    tmp0 = checkSize(tmp0, targ)

    diff ~ targ - pred  
    tmp0 ~ diff *@ diff
    lls ~ tmp0 * -1
    gradw ~ diff * 2
  }
}

class LogisticModel extends RegressionModel {
  class Options extends Learner.Options {}
  options = new Options
  
  options.blocksize = 10000
  options.npasses = 20
  options.alpha = 500f
  options.convslope = -1e-6
       
  override def regfn(targ:Mat, pred:Mat, lls:Mat, gradw:Mat):Unit =
    logisticMap1(targ.asInstanceOf[FMat], pred.asInstanceOf[FMat], lls.asInstanceOf[FMat], gradw.asInstanceOf[FMat])
  
  var tfact:FMat = null
  var ptfact:FMat = null
  var epred:FMat = null
  var lle:FMat = null
  var tmp0:FMat = null
  var tmp1:FMat = null
  
  def logisticMap1(targ:FMat, pred:FMat, lls:FMat, gradw:FMat):Unit= {
    tfact = checkSize(tfact, targ).asInstanceOf[FMat]
    ptfact = checkSize(ptfact, targ).asInstanceOf[FMat]
    epred = checkSize(epred, targ).asInstanceOf[FMat]
    lle = checkSize(lle, targ).asInstanceOf[FMat]
    
    var i = 0
    while (i < targ.length) {
      tfact.data(i) = 1-2*targ.data(i)
      ptfact.data(i) = math.min(40f, pred.data(i) * tfact.data(i))
      i+=1
    }
    exp(ptfact, epred)
    log1p(epred, lle)
    lls ~ row(-1) * lle
    i = 0
    while (i < targ.length) {
      gradw.data(i) = - tfact.data(i) * epred.data(i) / (1 + epred.data(i))
      i+=1
    }
    Mat.nflops += 8L * targ.length
  }
  
  def logisticMap2(targ:FMat, pred:FMat, lls:FMat, gradw:FMat) = {
    var i = 0
    while (i < targ.length) {
      val tfact:Double = (1-2*targ.data(i))
      val epred:Double = math.exp(math.min(40f, pred.data(i) * tfact))
      lls.data(i) = -math.log1p(epred).asInstanceOf[Float]
      gradw.data(i) = (- tfact * epred / (1 + epred)).asInstanceOf[Float]
      i += 1
    }
    Mat.nflops += 14L * targ.length
  }
    
  def logisticMap3(targ:FMat, pred:FMat, lls:FMat, gradw:FMat) = {
    tfact = checkSize(tfact, targ).asInstanceOf[FMat]
    epred = checkSize(epred, targ).asInstanceOf[FMat]
    tmp0 = checkSize(tmp0, targ).asInstanceOf[FMat]
    tmp1 = checkSize(tmp1, targ).asInstanceOf[FMat]
    
    tfact ~ row(1) - (tmp0 ~ row(2)*targ)
    min(40f, tmp0 ~ tfact *@ pred, tmp1)
    exp(tmp1, epred)
    log1p(epred, tmp0)
    lls ~ row(-1) * tmp0
    gradw ~ -1 * (tmp0 ~ tfact *@ (tmp1 ~ epred /@ (tmp0 ~ 1 + epred)))
  }
  
  def logisticMap4(targ:FMat, pred:FMat, lls:FMat, gradw:FMat) = {
    tfact = checkSize(tfact, targ).asInstanceOf[FMat]
    ptfact = checkSize(ptfact, targ).asInstanceOf[FMat]
    epred = checkSize(epred, targ).asInstanceOf[FMat]
    lle = checkSize(lle, targ).asInstanceOf[FMat]
    
    Learner.mapfun2x2((targ:Float, pred:Float) => (1-2*targ, math.min(40f, pred * (1-2*targ))), targ, pred, tfact, ptfact)
    exp(ptfact, epred)
    log1p(epred, lle)
    lls ~ row(-1) * lle
    Learner.mapfun2x1((tfact:Float, epred:Float)=>(-tfact*epred/(1+epred)), tfact, epred, gradw)
    Mat.nflops += 8L * targ.length
  }
}

abstract class RegressionModel extends Model {

  def regfn(targ:Mat, pred:Mat, lls:Mat, gradw:Mat):Unit
  
  def initmodel(learner:Learner, data:Mat, target:Mat):Mat = {
    val m = size(data, 1)
    val n = size(target, 1)
    val out = gnormrnd(0,1,n,m)*0.1f
//    val out = 0.1f*normrnd(0,1,m,n)
    println("norm="+norm(out))
    learner.modelmat = out
    out
  }
  
  var tpred:Mat = null
  var ttarget:Mat = null
  var lls:Mat = null
  var gradw:Mat = null
  var data:Mat = null
  var target:Mat = null
  
  def gradfun(idata:Mat, itarget:Mat, model:Mat, diff:Mat):Double = { 
    (idata, itarget, model) match {
  	  case (sdata:SMat, ftarget:FMat, gmodel:GMat) => {
  	  	lls = checkSize(lls, itarget.nrows, itarget.ncols, model)  	
  	  	tpred = checkSize(tpred, lls)
  	  	gradw = checkSize(gradw, lls)
  
  	    data = GSMat(sdata)
  	    target = GMat(ftarget)
  	    tpred ~ gmodel * data;
  	    regfn(target, tpred, lls, gradw)
  	    diff ~ gradw xT data
  	    target.asInstanceOf[GMat].free
  	    data.asInstanceOf[GSMat].free
  	    mean(mean(lls)).dv

  	  }
  	  case (sdata:SMat, ftarget:FMat, fmodel:FMat) => {
  	  	ttarget = checkSize(ttarget, itarget.ncols, itarget.nrows, itarget) 	
  	  	lls = checkSize(lls, ttarget)  	
  	  	tpred = checkSize(tpred, ttarget)
  	  	gradw = checkSize(gradw, ttarget)
  	  	
  	  	ttarget ~ ftarget t; 
  	  	tpred ~ sdata Tx fmodel 
  	  	regfn(ttarget, tpred, lls, gradw)
  	  	diff ~ sdata * gradw
  	  	mean(mean(lls)).dv
  	  }
    }
  } 
}

