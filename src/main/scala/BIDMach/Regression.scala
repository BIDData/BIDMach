package BIDMach

import BIDMat.{Mat,BMat,CMat,DMat,FMat,IMat,HMat,GMat,GIMat,GSMat,SMat,SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import Learner._

class LinearRegModel(data0:Mat, target0:Mat, opts:Model.Options = new Model.Options) extends RegressionModel(data0, target0, opts) {
  
  var tmp0:Mat = null
  var diff:Mat = null
  
  def regfn(targ:Mat, pred:Mat, lls:Mat, gradw:Mat):Unit =  linearMap1(targ, pred, lls, gradw)

  def linearMap1(targ:Mat, pred:Mat, lls:Mat, gradw:Mat):Unit= {
  	diff = checkSize(diff, targ)
    tmp0 = checkSize(tmp0, targ)

    diff ~ targ - pred  
    tmp0 ~ diff *@ diff
    lls ~ tmp0 * -1
    gradw ~ diff * 2
  }
  
  def linearMap2(targ:Mat, pred:Mat, lls:Mat, gradw:Mat):Unit= {
  	val ftarg = targ.asInstanceOf[FMat]
  	val fpred = pred.asInstanceOf[FMat]
  	val flls = lls.asInstanceOf[FMat]
  	val fgradw = gradw.asInstanceOf[FMat]
    
    var i = 0
    while (i < targ.length) {
      val diff = ftarg.data(i) - fpred.data(i)
      fgradw.data(i) = 2 * diff
      flls.data(i) = -diff * diff
      i += 1
    }
  	Mat.nflops += 4L*targ.length
  }
}

class LogisticModel(data0:Mat, target0:Mat, opts:Model.Options = new Model.Options) extends RegressionModel(data0, target0, opts)  {

  def regfn(targ:Mat, pred:Mat, lls:Mat, gradw:Mat):Unit = logisticMap1(targ, pred, lls, gradw)
  
  var tfact:Mat = null
  var ptfact:Mat = null
  var epred:Mat = null
  var lle:Mat = null
  var tmp0:Mat = null
  var tmp1:Mat = null
  
  var ftfact:FMat = null
  var fptfact:FMat = null
  var fepred:FMat = null
  var flle:FMat = null
  var ftmp0:FMat = null
  var ftmp1:FMat = null
  
  def logisticMap1(targ:Mat, pred:Mat, lls:Mat, gradw:Mat):Unit= {
    val ftarg = targ.asInstanceOf[FMat]
    val fpred = pred.asInstanceOf[FMat]
    val fgradw = gradw.asInstanceOf[FMat]
    ftfact = checkSize(tfact, ftarg).asInstanceOf[FMat]
    fptfact = checkSize(ptfact, ftarg).asInstanceOf[FMat]
    fepred = checkSize(epred, ftarg).asInstanceOf[FMat]
    flle = checkSize(lle, targ).asInstanceOf[FMat]
    
    var i = 0
    while (i < ftarg.length) {
      ftfact.data(i) = 1-2*ftarg.data(i)
      fptfact.data(i) = math.min(40f, fpred.data(i) * ftfact.data(i))
      i+=1
    }
    exp(fptfact, fepred)
    log1p(fepred, flle)
    lls ~ row(-1) * flle
    i = 0
    while (i < targ.length) {
      fgradw.data(i) = - ftfact.data(i) * fepred.data(i) / (1 + fepred.data(i))
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
    
  def logisticMap3(targ:Mat, pred:Mat, lls:Mat, gradw:Mat) = {
    tfact = checkSize(tfact, targ)
    epred = checkSize(epred, targ)
    tmp0 = checkSize(tmp0, targ)
    tmp1 = checkSize(tmp1, targ)
    
    tfact ~ 1 - (tmp0 ~ 2*targ)
    min(40.0, tmp0 ~ tfact *@ pred, tmp1)
    exp(tmp1, epred)
    log1p(epred, tmp0)
    lls ~ row(-1) * tmp0
    gradw ~ -1 * (tmp0 ~ tfact *@ (tmp1 ~ epred /@ (tmp0 ~ 1 + epred)))
  }
  
  def logisticMap4(targ:FMat, pred:FMat, lls:FMat, gradw:FMat) = {
    ftfact = checkSize(tfact, targ).asInstanceOf[FMat]
    fptfact = checkSize(ptfact, targ).asInstanceOf[FMat]
    fepred = checkSize(epred, targ).asInstanceOf[FMat]
    lle = checkSize(lle, targ).asInstanceOf[FMat]
    
    Learner.mapfun2x2((targ:Float, pred:Float) => (1-2*targ, math.min(40f, pred * (1-2*targ))), targ, pred, ftfact, fptfact)
    exp(fptfact, fepred)
    log1p(fepred, lle)
    lls ~ row(-1) * lle
    Learner.mapfun2x1((tfact:Float, epred:Float)=>(-tfact*epred/(1+epred)), ftfact, fepred, gradw)
    Mat.nflops += 8L * targ.length
  }
}

abstract class RegressionModel(data0:Mat, target0:Mat, opts:Model.Options) 
  extends Model(data0, target0, opts) {

  var modelmat:Mat = null
  var updatemat:Mat = null
  initmodel(data0, target0)

  def regfn(targ:Mat, pred:Mat, lls:Mat, gradw:Mat):Unit
  
  override def initmodel(data:Mat, target:Mat):Mat = initmodelf(data, target)
  
  def initmodelf(data:Mat, target:Mat):Mat = {
    val m = size(data, 1)
    val n = size(target, 1)
    modelmat = 0.1f*normrnd(0,1,m,n)
    updatemat = modelmat.zeros(m, n)
    target
  }
  
  def initmodelg(data:Mat, target:Mat):Mat = {
    val m = size(data, 1)
    val n = size(target, 1)
    modelmat = gnormrnd(0,1,n,m)*0.1f
    updatemat = modelmat.zeros(n, m)
    target
  }
  
  var tpred:Mat = null
  var ttarget:Mat = null
  var lls:Mat = null
  var gradw:Mat = null
  var data:Mat = null
  var target:Mat = null
  
  override def gradfun(idata:Mat, itarget:Mat):Double = { 
    (idata, itarget, modelmat) match {
  	  case (sdata:SMat, ftarget:FMat, gmodel:GMat) => {
  	  	lls = checkSize(lls, itarget.nrows, itarget.ncols, modelmat)  	
  	  	tpred = checkSize(tpred, lls)
  	  	gradw = checkSize(gradw, lls)
  
  	    data = GSMat(sdata)
  	    target = GMat(ftarget)
  	    tpred ~ gmodel * data;
  	    regfn(target, tpred, lls, gradw)
  	    updatemat ~ gradw xT data
  	    target.asInstanceOf[GMat].free
  	    data.asInstanceOf[GSMat].free
  	    mean(mean(lls,2)).dv
  	  }
  	  case (sdata:SMat, ftarget:FMat, fmodel:FMat) => {
  	  	ttarget = checkSize(ttarget, itarget.ncols, itarget.nrows, itarget) 	
  	  	lls = checkSize(lls, ttarget)  	
  	  	tpred = checkSize(tpred, ttarget)
  	  	gradw = checkSize(gradw, ttarget) 	  	
  	  	ttarget ~ ftarget t; 
  	  	tpred ~ sdata Tx fmodel 
  	  	regfn(ttarget, tpred, lls, gradw)
  	  	updatemat ~ sdata * gradw
  	  	mean(mean(lls,1)).dv
  	  }
    }
  } 
}

