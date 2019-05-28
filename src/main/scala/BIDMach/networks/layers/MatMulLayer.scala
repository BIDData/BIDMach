package BIDMach.networks.layers

import BIDMat.{Mat,SBMat,CMat,DMat,FMat,IMat,LMat,HMat,GMat,GDMat,GIMat,GLMat,GSMat,GSDMat,SMat,SDMat}
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
 * Computes the matrix product of two input layers. 
 * For higher dimensional tensors (> 2), the first two dimensions determine the matrix product dimensions.
 * The matrix multiply is repeated over the other dimensions. 
 */

@SerialVersionUID(100L)
class MatMulLayer(override val net:Net, override val opts:MatMulNodeOpts = new MatMulNode) extends Layer(net, opts) {  
  
	override val _inputs = new Array[LayerTerm](2);

	override def forward = {
    val start = toc;
    val coli = if (opts.skipDim1) 2 else 1;
    val nrows = if (opts.transA) inputData.dims(coli) else inputData.dims(0);
    val ncols = if (opts.transB) inputDatas(1).dims(0) else inputDatas(1).dims(coli);
    val nblocks = if (inputData.dims.length <= 2) {
    	createOutput(nrows \ ncols);
    	1
    } else {
      if (!opts.skipDim1) {
    	  createOutput(nrows \ ncols \ inputData.dims(2->inputData.dims.length));
    	  inputData.dims.data.slice(2, inputData.dims.length).reduce(_*_)
      } else {
        createOutput(nrows \ inputData.dims(1) \ ncols \ inputData.dims(3->inputData.dims.length));
        inputData.dims(1)
      }
    }
    inplaceNoConnectGetOutput();
	        
    if (!opts.skipDim1) {
    	inputData.blockmult(inputDatas(1), output, nblocks, opts.transA, opts.transB)
    } else {
      inputData.blockmult2(inputDatas(1), output, nblocks, opts.transA, opts.transB)    
    }
	
	  forwardtime += toc - start;
	}

	override def backward = {
    val start = toc;
    inplaceNoConnectGetInputDerivs();
     val nblocks = if (inputData.dims.length <= 2) {
      1
    } else {
      if (!opts.skipDim1) {
        inputData.dims.data.slice(2, inputData.dims.length).reduce(_*_)
      } else {
        inputData.dims(1)
      }
    }
    
    if (inputDerivs(0).asInstanceOf[AnyRef] != null) {
      if (!opts.skipDim1) {
    	  if (! opts.transA) {
    		  deriv.blockmadd(inputDatas(1), inputDerivs(0), nblocks, false, ! opts.transB)      
    	  } else {
    		  inputDatas(1).blockmadd(deriv, inputDerivs(0), nblocks, opts.transB, true)
    	  }
      } else {
        if (! opts.transA) {
          deriv.blockmadd2(inputDatas(1), inputDerivs(0), nblocks, false, ! opts.transB)      
        } else {
          inputDatas(1).blockmadd2(deriv, inputDerivs(0), nblocks, opts.transB, true)
        }        
      }
    }
    if (inputDerivs(1).asInstanceOf[AnyRef] != null) {
    	if (!opts.skipDim1) {
    		if (! opts.transB) {
    			inputData.blockmadd(deriv, inputDerivs(1), nblocks, ! opts.transA, false)
    		} else {
    			deriv.blockmadd(inputData, inputDerivs(1), nblocks, true, opts.transA)
    		}
      } else {
        if (! opts.transB) {
          inputData.blockmadd2(deriv, inputDerivs(1), nblocks, ! opts.transA, false)
        } else {
          deriv.blockmadd2(inputData, inputDerivs(1), nblocks, true, opts.transA)
        }      
      }
    }
    
    inplaceNoConnectReleaseDeriv()
    backwardtime += toc - start;
	}
  
  override def toString = {
    "matmul@"+Integer.toHexString(hashCode % 0x10000).toString
  }
}

trait MatMulNodeOpts extends NodeOpts {  
  var transA = false
  var transB = false
  var skipDim1 = false

  def copyOpts(opts:MatMulNodeOpts):MatMulNodeOpts = {
      super.copyOpts(opts);
      opts.transA = transA
      opts.transB = transB
      opts;
  }
}

@SerialVersionUID(100L)
class MatMulNode extends Node with MatMulNodeOpts {
  override val inputs:Array[NodeTerm] = new Array[NodeTerm](2);
  
  def copyTo(opts:MatMulNode):MatMulNode = {
    super.copyTo(opts);
    copyOpts(opts);
    opts;
  }

	override def clone:MatMulNode = {copyTo(new MatMulNode).asInstanceOf[MatMulNode];}

	override def create(net:Net):MatMulLayer = {MatMulLayer(net, this);}
  
  override def toString = {
    "matmul@"+Integer.toHexString(hashCode % 0x10000).toString
  }
}
  
@SerialVersionUID(100L)
object MatMulLayer {  
  
  def apply(net:Net) = new MatMulLayer(net, new MatMulNode);
  
  def apply(net:Net, opts:MatMulNodeOpts) = new MatMulLayer(net, opts); 
}
