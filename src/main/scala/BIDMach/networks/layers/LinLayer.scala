package BIDMach.networks.layers

import BIDMat.{Mat,SBMat,CMat,DMat,FMat,IMat,LMat,HMat,GMat,GDMat,GIMat,GLMat,GSMat,GSDMat,SMat,SDMat,TMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import BIDMach.datasources._
import BIDMach.updaters._
import BIDMach.mixins._
import BIDMach.models._
import BIDMach._
import edu.berkeley.bid.CPUMACH
import edu.berkeley.bid.CUMACH
import scala.util.hashing.MurmurHash3;
import java.util.HashMap;
import BIDMach.networks._ 

/**
 * Linear layer. 
 * Includes a model matrix that contains the linear map. 
 */

class LinLayer(override val net:Net, override val opts:LinNodeOpts = new LinNode) extends ModelLayer(net, opts) {
  var vexp:Mat = null;
  var texp:Mat = null;
  var lrate:Mat = null;
//  var sumsq:Mat = null;
  var mask:Mat = null;
  var dprod:Mat = null;
  var firststep = -1f;
  var waitsteps = 0;
  var epsilon = 0f;
  var ADAinitialized = false;
  
  def initModelMat(nr:Int, nc2:Int):Mat = {
    println(imodel,opts.modelName)
    if (imodel != 0 && imodel<9) {
    //if (imodel != 0 && imodel<3) 
    //if (true)
        val nc = nc2
        (rand(nr, nc) - 0.5f)*0.15
    }
    else{
        val nc = nc2 - 1
        opts.hasBias = false
        //val x=Array(0,105,113)
        //println("Create",nr,nc)
        ///println(GPUmem._1)
        val n = 4
        //val x=(0 until n).toArray.map(nc*_/n)//Array(0,nc/3,nc/3*2)
        //val x=Array(0,20000,40000,60000)
        val x=Array(0,1000,10000,20000)
        val ratio = 4
        //val y=Array(nr/2,nr/4,0)//nr/2)
        //val y=(0 until n).toArray.map((nr-nr/ratio)*_/(n-1))
        //val y=Array(0,64,128,224)
        val y=Array(0,1024,1024+512,1024+512+256)
        //val r=Array(nr/4*3,nr/4*3,nr/4*3)
        val nnr=if (nr<nc)nr else nc
        val nnc=if (nr<nc)nc else nr
        val r=y.map(nnr-_)
        val c=(0 until n).toArray.map(i=>(if (i!=n-1) x(i+1) else nnc)-x(i))
        //TMat(nr,nc,x,y,Array((GMat(rand(nr, nc)) - 0.5f)*0.15));
        val res = if (nr<nc)
                    TMat(nr,nc,x,y,(0 until n).toArray.map(i=>(GMat(rand(r(i), c(i))) - 0.5f)*0.15))
                  else
                    TMat(nr,nc,y,x,(0 until n).toArray.map(i=>(GMat(rand(c(i), r(i))) - 0.5f)*0.15))
        //println(GPUmem._1)
        res
    }
  }
  
    var tot=0
  override def forward = {
     // println(opts.modelName,GPUmem._1)
    
    val start = toc;
    val modelcols = inputData.nrows;
    if (modelmats(imodel).asInstanceOf[AnyRef] == null) {
      val outdim = if (opts.outdim == 0) inputData.nrows else opts.outdim;
      modelmats(imodel) = convertMat(initModelMat(outdim, modelcols + (if (opts.hasBias) 1 else 0)));
      updatemats(imodel) = modelmats(imodel).zeros(modelmats(imodel).nrows, modelmats(imodel).ncols);  
      updatemats(imodel).clear
    }else
    if (modelmats(imodel).isInstanceOf[TMat])
        opts.hasBias = false
    
    if (opts.aopts != null && !ADAinitialized) initADAGrad;
    val mm = if (opts.hasBias) modelmats(imodel).view(modelmats(imodel).nrows, modelcols) else modelmats(imodel);
    ///if (imodel ==6)println("In f0",GPUmem._1)
    createOutput(mm.nrows \ inputData.ncols);
    //println("!")
    //if (imodel ==6)println("In f1",GPUmem._1)
    output.asMat ~ mm * inputData.asMat;
    //println("@@")
    //if (imodel ==6)println("In f2",GPUmem._1)
    if (opts.hasBias) output.asMat ~ output.asMat + modelmats(imodel).colslice(modelcols, modelcols+1);
    clearDeriv;
    forwardtime += toc - start;
    //if (imodel ==6)println("Out f",GPUmem._1)
    ///println(opts.modelName,GPUmem._1)
    
  }

  override def backward(ipass:Int, pos:Long) = {

    val start = toc;
	  val modelcols = inputData.nrows;
    val mm = if (opts.hasBias) modelmats(imodel).view(modelmats(imodel).nrows, modelcols) else modelmats(imodel);
    if (inputDeriv.asInstanceOf[AnyRef] != null) {
      mm.madd(deriv.asMat, inputDeriv.asMat, true, false);
    }
    if (opts.aopts != null) {
      if (firststep <= 0) firststep = pos.toFloat;
      val istep = (pos + firststep)/firststep;
      ADAGrad.multUpdate(deriv.asMat, inputData.asMat, modelmats(imodel), updatemats(imodel), mask, lrate, texp, vexp, epsilon, istep, waitsteps);
    } else {
        val um = if (opts.hasBias) updatemats(imodel).view(updatemats(imodel).nrows, modelcols) else updatemats(imodel);
        deriv.asMat.madd(inputData.asMat, um, false, true);
      if (opts.hasBias) updatemats(imodel)(?,modelcols) = updatemats(imodel)(?,modelcols) + sum(deriv.asMat,2)
    }
    backwardtime += toc - start;

  }


  def initADAGrad {
    val aopts = opts.aopts;
    val mm = modelmats(imodel); 
    val d = mm.nrows;
    val m = mm.ncols;
    firststep = -1f;
    lrate = convertMat(aopts.lrate);
    texp = convertMat(aopts.texp);
    vexp = convertMat(aopts.vexp);
//    sumsq = convertMat(zeros(d, m));
    updatemats(imodel).set(aopts.initsumsq);
    waitsteps = aopts.waitsteps;
    epsilon = aopts.epsilon;
    mask = aopts.mask;
    ADAinitialized = true;
  }
  
  override def toString = {
    "linear@"+Integer.toHexString(hashCode % 0x10000).toString
  }
}

trait LinNodeOpts extends ModelNodeOpts {
	var hasBias:Boolean = false;
  var aopts:ADAGrad.Opts = null;
  var outdim = 0;
  
  def copyOpts(opts:LinNodeOpts):LinNodeOpts = {
  		super.copyOpts(opts);
  		opts.hasBias = hasBias;
  		opts.aopts = aopts;
  		opts.outdim = outdim;
  		opts;
  }
}
    
class LinNode extends ModelNode with LinNodeOpts {
  def copyTo(opts:LinNode):LinNode = {
    this.asInstanceOf[Node].copyTo(opts);
    copyOpts(opts);
    opts
  }
  
  override def toString = {
    "linear@"+Integer.toHexString(hashCode % 0x10000).toString
  }
    
  override def clone:LinNode = {
    copyTo(new LinNode).asInstanceOf[LinNode];
  }
  
  override def create(net:Net):LinLayer = {
  	LinLayer(net, this);
  }
}

object LinLayer {  

  def apply(net:Net) = new LinLayer(net, new LinNode);
  
  def apply(net:Net, opts:LinNodeOpts):LinLayer = new LinLayer(net, opts);
  
}