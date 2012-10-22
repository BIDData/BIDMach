package BIDMach
import BIDMat.{Mat,BMat,CMat,DMat,FMat,IMat,HMat,GMat,GIMat,GSMat,SMat,SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import Learner._

class LinearRegModel extends RegressionModel {
  class Options extends Learner.Options {}
  options = new Options
  
  options.blocksize = 10000
  options.npasses = 100
  options.alpha = 200f
  options.convslope = -1e-6
  
  var diff:FMat = null
  var tmp0:FMat = null
       
  override def regfn(targ:FMat, pred:FMat, lls:FMat, gradw:FMat):Unit =
    linearMap1(targ, pred, lls, gradw)
  
  def linearMap1(targ:FMat, pred:FMat, lls:FMat, gradw:FMat):Unit= {
    diff = checkSize(diff, targ)
    tmp0 = checkSize(tmp0, targ)
    
    diff ~ targ - pred  
    lls ~ (tmp0 ~ diff *@ diff) * -1
    gradw ~ diff * 2
  }
}

class LogisticModel extends RegressionModel {
  class Options extends Learner.Options {}
  options = new Options
  
  options.blocksize = 10000
  options.npasses = 100
  options.alpha = 500f
  options.convslope = -1e-6
  
  var tfact:FMat = null
  var ptfact:FMat = null
  var epred:FMat = null
  var lle:FMat = null
  var tmp0:FMat = null
  var tmp1:FMat = null
  
  override def initmodel(learner:Learner, data:Mat, target:Mat):Mat = {
    val out = super.initmodel(learner, data, target)
    out
  }
      
  override def regfn(targ:FMat, pred:FMat, lls:FMat, gradw:FMat):Unit =
    logisticMap1(targ, pred, lls, gradw)
  
  def logisticMap1(targ:FMat, pred:FMat, lls:FMat, gradw:FMat):Unit= {
    tfact = checkSize(tfact, targ)
    ptfact = checkSize(ptfact, targ)
    epred = checkSize(epred, targ)
    lle = checkSize(lle, targ)
    
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
    tfact = checkSize(tfact, targ)
    epred = checkSize(epred, targ)
    tmp0 = checkSize(tmp0, targ)
    tmp1 = checkSize(tmp1, targ)
    
    tfact ~ row(1) - (tmp0 ~ row(2)*targ)
    min(40f, tmp0 ~ tfact *@ pred, tmp1)
    exp(tmp1, epred)
    log1p(epred, tmp0)
    lls ~ row(-1) * tmp0
    gradw ~ -1 * (tmp0 ~ tfact *@ (tmp1 ~ epred /@ (tmp0 ~ 1 + epred)))
  }
  
  def logisticMap4(targ:FMat, pred:FMat, lls:FMat, gradw:FMat) = {
    tfact = checkSize(tfact, targ)
    ptfact = checkSize(ptfact, targ)
    epred = checkSize(epred, targ)
    lle = checkSize(lle, targ)
    
    Learner.mapfun2x2((targ:Float, pred:Float) => (1-2*targ, math.min(40f, pred * (1-2*targ))), targ, pred, tfact, ptfact)
    exp(ptfact, epred)
    log1p(epred, lle)
    lls ~ row(-1) * lle
    Learner.mapfun2x1((tfact:Float, epred:Float)=>(-tfact*epred/(1+epred)), tfact, epred, gradw)
    Mat.nflops += 8L * targ.length
  }
}

abstract class RegressionModel extends Model {
    
  var tpred:FMat = null
  var fttarget:FMat = null
  var lls:FMat = null
  var gradw:FMat = null
  var modelmat:FMat = null

  def regfn(targ:FMat, pred:FMat, lls:FMat, gradw:FMat):Unit
  
  def initmodel(learner:Learner, data:Mat,target:Mat):Mat = {
    val m = size(data, 1)
    val n = size(target, 1)
    val out = 0.1f*normrnd(0,1,m,n)
    learner.modelmat = out
    modelmat = out
    out
  }
  
  def gradfun(data:Mat, target:Mat, model:Mat, diff:Mat):Double = {
  	val sdata = data.asInstanceOf[SMat]
  	val ftarget = target.asInstanceOf[FMat]
  	val fmodel = model.asInstanceOf[FMat]
  	val fdiff = diff.asInstanceOf[FMat]
  	
  	fttarget = checkSize(fttarget, target.ncols, target.nrows)
  	lls = checkSize(lls, fttarget)
  	tpred = checkSize(tpred, fttarget)
  	gradw = checkSize(gradw, fttarget)
  	
  	fttarget ~ ftarget t;
  	tpred ~ sdata Tx fmodel 
  	regfn(fttarget, tpred, lls, gradw)
  	fdiff ~ sdata * gradw
    mean(mean(lls)).v
  }
}

