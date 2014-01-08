package BIDMach.models

import BIDMat.{Mat,BMat,CMat,DMat,FMat,IMat,HMat,GMat,GIMat,GSMat,SMat,SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
//import edu.berkeley.bid.CUMAT

import BIDMach.datasources._
import BIDMach.updaters._
import BIDMach._


class GibbsLDAModel(override val opts:GibbsLDAModel.Opts = new GibbsLDAModel.Options) extends FactorModel(opts) { 
  var mm:Mat = null
  var alpha:Mat = null
  
  var traceMem = false
  
  override def init(datasource:DataSource) = {
    super.init(datasource)
    mm = modelmats(0)
    modelmats = new Array[Mat](2)
    modelmats(0) = mm
    modelmats(1) = mm.ones(mm.nrows, 1)
    updatemats = new Array[Mat](2)
    updatemats(0) = mm.zeros(mm.nrows, mm.ncols)
    updatemats(1) = mm.zeros(mm.nrows, 1)
    

  }
  
  // use case match to handle GMat and etc.
  def uupdate(sdata:Mat, user:Mat, ipass: Int):Unit = {
    
    	if (opts.putBack < 0 || ipass == 0) user.set(1f)
        for (i <- 0 until opts.uiter) {
    	val preds = DDS(mm, user, sdata)	
    	if (traceMem) println("uupdate %d %d %d, %d %f %d" format (mm.GUID, user.GUID, sdata.GUID, preds.GUID, GPUmem._1, getGPU))
    	val dc = sdata.contents
    	val pc = preds.contents
    	pc ~ pc / dc
    	
    	val unew = user*0
    	val mnew = updatemats(0)
    	//mnew.set(0f)
  
    	//CUMAT.LDAgibbs(opts.dim, sdata.asInstanceOf[GSMat].nnz, mm.asInstanceOf[GMat].data, user.asInstanceOf[GMat].data, updatemats(0).asInstanceOf[GMat].data, unew.asInstanceOf[GMat].data, sdata.asInstanceOf[GSMat].ir, sdata.asInstanceOf[GSMat].ic, pc.asInstanceOf[GMat].data, opts.nsamps)
    	LDAgibbs(mm, user, mnew, unew, preds, opts.nsamps)
        
    	if (traceMem) println("uupdate %d %d %d, %d %d %d %d %f %d" format (mm.GUID, user.GUID, sdata.GUID, preds.GUID, dc.GUID, pc.GUID, unew.GUID, GPUmem._1, getGPU))
    	user ~ unew + opts.alpha
    	}
  
  }
  
  def mupdate(sdata:Mat, user:Mat, ipass: Int):Unit = {
	val um = updatemats(0)
	um ~ um + opts.beta 
  	sum(um, 2, updatemats(1))
  }
  
  def evalfun(sdata:Mat, user:Mat):FMat = {  
  	val preds = DDS(mm, user, sdata)
  	val dc = sdata.contents
  	val pc = preds.contents
  	max(opts.weps, pc, pc)
  	ln(pc, pc)
  	val sdat = sum(sdata,1)
  	val mms = sum(mm,2)
  	val suu = ln(mms ^* user)
  	if (traceMem) println("evalfun %d %d %d, %d %d %d, %d %f" format (sdata.GUID, user.GUID, preds.GUID, pc.GUID, sdat.GUID, mms.GUID, suu.GUID, GPUmem._1))
  	val vv = ((pc ddot dc) - (sdat ddot suu))/sum(sdat,2).dv
  	row(vv, math.exp(-vv))
  }
}

object GibbsLDAModel  {
  
  trait Opts extends FactorModel.Opts {
    var alpha = 0.001f
    var beta = 0.0001f
    var nsamps = 1
  }
  
  class Options extends Opts {}
  
  def mkGibbsLDAmodel(fopts:Model.Opts) = {
  	new GibbsLDAModel(fopts.asInstanceOf[GibbsLDAModel.Opts])
  }
  
  def mkUpdater(nopts:Updater.Opts) = {
  	new IncNormUpdater(nopts.asInstanceOf[IncNormUpdater.Opts])
  } 
  
  
  def learn(mat0:Mat, d:Int = 256) = {
    class xopts extends Learner.Options with GibbsLDAModel.Opts with MatDataSource.Opts with IncNormUpdater.Opts
    val opts = new xopts
    opts.dim = d
    opts.putBack = 1
    opts.blockSize = math.min(100000, mat0.ncols/30 + 1)
  	val nn = new Learner(
  	    new MatDataSource(Array(mat0:Mat), opts), 
  			new GibbsLDAModel(opts), 
  			null, 
  			new IncNormUpdater(opts), opts)
    (nn, opts)
  }
  
}


