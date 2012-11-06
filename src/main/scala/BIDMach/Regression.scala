package BIDMach

import BIDMat.{Mat,BMat,CMat,DMat,FMat,IMat,HMat,GMat,GIMat,GSMat,SMat,SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import Learner._

class LinearRegModel(data0:Mat, target0:Mat, opts:RegressionModel.Options = new RegressionModel.Options) 
extends RegressionModel(data0, target0, opts) {
  
  var tmp0:Mat = null
  var diff:Mat = null
  
  def regfn(targ:Mat, pred:Mat, lls:Mat, gradw:Mat):Unit =  linearMap1(targ, pred, lls, gradw)

  def linearMap1(targ:Mat, pred:Mat, lls:Mat, gradw:Mat):Unit= {
  	diff = recycleTry(diff, targ)
    tmp0 = recycleTry(tmp0, targ)

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

class LogisticModel(data0:Mat, target0:Mat, opts:RegressionModel.Options = new RegressionModel.Options) 
extends RegressionModel(data0, target0, opts)  {

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
    ftfact = recycleTry(tfact, ftarg).asInstanceOf[FMat]
    fptfact = recycleTry(ptfact, ftarg).asInstanceOf[FMat]
    fepred = recycleTry(epred, ftarg).asInstanceOf[FMat]
    flle = recycleTry(lle, targ).asInstanceOf[FMat]
    
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
    tfact = recycleTry(tfact, targ)
    epred = recycleTry(epred, targ)
    tmp0 = recycleTry(tmp0, targ)
    tmp1 = recycleTry(tmp1, targ)
    
    tfact ~ 1 - (tmp0 ~ 2*targ)
    min(40.0, tmp0 ~ tfact *@ pred, tmp1)
    exp(tmp1, epred)
    log1p(epred, tmp0)
    lls ~ row(-1) * tmp0
    gradw ~ -1 * (tmp0 ~ tfact *@ (tmp1 ~ epred /@ (tmp0 ~ 1 + epred)))
  }
  
  def logisticMap4(targ:FMat, pred:FMat, lls:FMat, gradw:FMat) = {
    ftfact = recycleTry(tfact, targ).asInstanceOf[FMat]
    fptfact = recycleTry(ptfact, targ).asInstanceOf[FMat]
    fepred = recycleTry(epred, targ).asInstanceOf[FMat]
    lle = recycleTry(lle, targ).asInstanceOf[FMat]
    
    Learner.mapfun2x2((targ:Float, pred:Float) => (1-2*targ, math.min(40f, pred * (1-2*targ))), targ, pred, ftfact, fptfact)
    exp(fptfact, fepred)
    log1p(fepred, lle)
    lls ~ row(-1) * lle
    Learner.mapfun2x1((tfact:Float, epred:Float)=>(-tfact*epred/(1+epred)), ftfact, fepred, gradw)
    Mat.nflops += 8L * targ.length
  }
}

abstract class RegressionModel(data0:Mat, target0:Mat, opts:RegressionModel.Options) 
  extends Model(data0, target0, opts) {
  
  val options = opts

  var modelmat:Mat = null
  var updatemat:Mat = null    
  var data:Mat = null
  var target:Mat = null
  var ttarget:Mat = null
  var lls:Mat = null
  var tpred:Mat = null
  var gradw:Mat = null

  def regfn(targ:Mat, pred:Mat, lls:Mat, gradw:Mat):Unit
  
  override def initmodel(data:Mat, target:Mat):Mat = initmodelf(data, target)
  
  def initmodelf(data:Mat, target:Mat) = {
    val m = size(data, 1)
    val n = size(target, 1)
    if (options.transpose) {
    	modelmat = 0.1f*normrnd(0,1,m,n)
    	updatemat = modelmat.zeros(m,n)
    } else {
    	modelmat = 0.1f*normrnd(0,1,n,m)
    	updatemat = modelmat.zeros(n,m)
    }
    target
  }
  
  def initmodelg(data0:Mat, target0:Mat):Mat = {
    val m = size(data0, 1)
    val n = size(target0, 1)
    val k = options.startBlock
    if (options.transpose) {
    	modelmat = gnormrnd(0,1,m,n)*0.1f
    	updatemat = modelmat.zeros(m,n)
    } else {
    	modelmat = gnormrnd(0,1,n,m)*0.1f
    	updatemat = modelmat.zeros(n, m)
    }
    data = GSMat(m, k, k * options.nzPerColumn)
    target = GMat(n, k)
    target
  }
  
  override def gradfun(idata:Mat, itarget:Mat):Double = { 
    if (options.transpose) {
    	ttarget = recycleTry(ttarget, itarget.ncols, itarget.nrows, itarget) 
    	ttarget ~ itarget t;
    } else {
    	ttarget = itarget
    }
    lls = recycleTry(lls, ttarget.nrows, ttarget.ncols, modelmat)
    tpred = recycleTry(tpred, lls)
    gradw = recycleTry(gradw, lls)
    modelmat match {
    case m:GMat => {
    	data = GSMat.fromSMat(idata.asInstanceOf[SMat], data.asInstanceOf[GSMat])
    	target = GMat.fromFMat(itarget.asInstanceOf[FMat], target.asInstanceOf[GMat])
    }
    case m:FMat => {
      data = idata
      target = ttarget 
    }
    }
    if (options.transpose) {
      tpred ~ data Tx modelmat
      regfn(target, tpred, lls, gradw)
      updatemat ~ data * gradw
      mean(mean(lls,1)).dv 
    } else {
    	tpred ~ modelmat * data;
    	regfn(target, tpred, lls, gradw)
    	updatemat ~ gradw xT data
    	mean(mean(lls,2)).dv
    }	                   
  } 
}

object RegressionModel {
  class Options extends Model.Options {
    var transpose = true
  }
}
