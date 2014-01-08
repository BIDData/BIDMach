package BIDMach.models

import BIDMat.{Mat,BMat,CMat,DMat,FMat,IMat,HMat,GMat,GIMat,GSMat,SMat,SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
//import edu.berkeley.bid.CUMAT

import BIDMach.datasources._
import BIDMach.updaters._


class GibbsLDAModel(override val opts:GibbsLDAModel.Options = new GibbsLDAModel.Options) extends FactorModel(opts) { 
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
    	mnew.set(0f)
  
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
  class Options extends FactorModel.Options {
    var LDAeps = 1e-9
    var alpha = 0.001f
    var beta = 0.0001f
    var nsamps = 1
  }
  
}


/*
class GibbsLDAModel(sdata: SMat, k: Int, nsamps: Float, w: Float, alpha: Float, beta: Float, sbatch: Int, niter: Int) {
	
  val (nfeats, nusers) = size(sdata)
  val nbatch = nusers/sbatch
  
  var A: GMat = null
  //var B: GMat = null
  var Bs: Array[GMat] = Array.fill[GMat](nbatch)(null)
  var AN: GMat = null
  var BN: GMat = null
  //var bdata:GSMat = null

  
  def init = {
    
    A = grand(k, nfeats) + alpha
    for(j <- 0 until nbatch){
    	Bs(j) = grand(k, sbatch) + beta
    }
    AN = gzeros(k, nfeats)
    BN = gzeros(k, sbatch)
  }
  /*
  def update = {
    
    // iteration
    for (i <- 0 until 10) {
      
      //mini-batch
      for(j <- 0 until nbatch){  
    	val bdata = GSMat(sdata(?, j*sbatch until (j+1)*sbatch))
        val preds = DDS(A, B, bdata)	
        val dc = bdata.contents
	  	val pc = preds.contents
	  	//max(1e-6f, pc, pc)
	  	//pc ~ dc / pc
        pc ~ pc / dc
    	LDAgibbs(k, bdata.nnz, A.data, B.data, AN.data, BN.data, bdata.ir, bdata.ic, pc.data, nsamps)
        //A = w*A + (1-w)*AN + alpha
    	A += AN + alpha
    	val suma = sum(A)
    	A /= suma
        B = BN + beta
        val sumb = sum(B)
        B /= sumb
        AN.clear
        BN.clear
        println("iteration: %d, batch: %d, perplexity: %f".format(i, j, perplexity(bdata)))
      }
      //println("iteration: %d, perplexity: %f".format(i, perplexity))
    }
  }
  */
   def update2 = {
    
    // iteration
    for (i <- 0 until niter) {
      
      //mini-batch
      for(j <- 0 until nbatch){  
    	val bdata = GSMat(sdata(?, j*sbatch until (j+1)*sbatch))
        val preds = DDS(A, Bs(j), bdata)	
        val dc = bdata.contents
	  	val pc = preds.contents
	  	//max(1e-6f, pc, pc)
	  	//pc ~ dc / pc
        pc ~ pc / dc
    	LDAgibbs(k, bdata.nnz, A.data, Bs(j).data, AN.data, BN.data, bdata.ir, bdata.ic, pc.data, nsamps)
        
    	//val wA = w*A
    	//val update = (1-w)*(AN + alpha) 
    	
    	A ~ A * w
    	AN ~ AN + alpha
    	AN ~ AN * (1-w)
    	A ~ A + AN 
    	//A ~ A + alpha
    	val suma = sum(A)
    	A ~ A / suma
  
        Bs(j) ~ BN + beta
        AN.clear
        BN.clear
        //if(j==0){
        	println("iteration: %d, batch: %d, perplexity: %f".format(i, j, perplexity(bdata, A, Bs(j))))
        //}
      }
      //println("iteration: %d, perplexity: %f".format(i, perplexity()))
    }
  }
  
  def perplexity(bdata: GSMat, A: GMat, B: GMat):Double = {  
    //A = A / sum(A)
    //B = B / sum(B)
  	val preds = DDS(A, B, bdata)
  	val dc = bdata.contents
  	val pc = preds.contents
  	max(1e-6f, pc, pc)
  	ln(pc, pc)
  	val sdat = sum(bdata,1)
  	val mms = sum(A,2)
  	val suu = ln(mms ^* B) 
  	val vv = ((pc ddot dc) - (sdat ddot suu))/sum(sdat,2).dv
  	//row(vv, math.exp(-vv))
  	math.exp(-vv)
  }
  

}

object GibbsLDAModel{
  
  def loadFile(fname: String, oname: String)={
    val f = new BufferedReader(new FileReader(fname))
    val D = f.readLine().toInt
    val W = f.readLine().toInt
    val nnz = f.readLine().toInt
    
    var r = izeros(nnz, 1)
    var c = izeros(nnz, 1)
    var v = zeros(nnz, 1)
    
    var i = 0
    
    var l = f.readLine()
    while(l != null){
      var t = l.split(" ")
      r(i) = t(0).toInt-1
      c(i) = t(1).toInt-1
      v(i) = t(2).toFloat
      i = i+1
      l = f.readLine()
    }
    
    //val smat = sparse(r(0 to i-1), c(0 to i-1), v(0 to i-1))
    val smat = sparse(c(0 to i-1), r(0 to i-1), v(0 to i-1))
    
    saveSMat(oname, smat)
  }
  
  def main(args: Array[String]) = {
	val fname = args(0)
	
	//val oname = args(1)
	//loadFile(fname, oname)
	
	
	val k = args(1).toInt
	val nsamps = args(2).toFloat
	val w = args(3).toFloat
	val alpha = args(4).toFloat
	val beta = args(5).toFloat
	val sbatch = args(6).toInt
	val niter = args(7).toInt
	// sdata dimension nfeats (words) * nusers (documents)
	val data:SMat = loadSMat(fname)
	println("size of smat: " + size(data))
	println("nnz of smat: " + data.nnz)
	
	Mat.checkMKL
	Mat.checkCUDA

	println("gpu: " + Mat.hasCUDA)
    //val sdata = GSMat(data)
    
    val model = new GibbsLDAModel(data, k, nsamps, w, alpha, beta, sbatch, niter)
	model.init
	model.update2
	
	
    
  }
  
}*/