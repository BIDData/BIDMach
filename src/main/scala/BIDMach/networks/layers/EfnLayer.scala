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
 *  Element-wise unary function layer. Converts input to an FMat, runs the function over the data, and
 *  returns the output for each element. 
 *  
 */

class EfnLayer(override val net:Net, override val opts:EfnNodeOpts = new EfnNode) extends Layer(net, opts) {
  var finput:FMat = null;
  var foutput:FMat = null;
  var finderiv:FMat = null;
  var fderiv:FMat = null;

  override def forward = {
		  val start = toc;
		  inplaceNoConnectGetOutput();
		        
		  if (finput.asInstanceOf[AnyRef] == null) {
		    finput = FMat(inputData);
		    foutput = FMat(output);		    
		  }
		  if (opts.fwdfn.asInstanceOf[AnyRef] != null) {
		  	finput <-- inputData;
		  	var i = 0;
		  	while (i < finput.length) {
		  	  foutput.data(i) = opts.fwdfn(finput.data(i));
		  	  i += 1;
		  	}
		  	output <-- foutput;
		  }
		  forwardtime += toc - start;
  }

  override def backward = {
		  val start = toc;
		  inplaceNoConnectGetInputDerivs();
		  
		  if (inputDeriv.asInstanceOf[AnyRef] != null &&  opts.bwdfn.asInstanceOf[AnyRef] != null) {
		  	if (finderiv.asInstanceOf[AnyRef] == null) {
		  		finderiv = FMat(inputDeriv);
		  		fderiv = FMat(deriv);		    
		  	}
		  	finput <-- inputData;
		  	foutput <-- output;
		  	fderiv <-- deriv;
		  	var i = 0;
		  	while (i < finput.length) {
		  	  finderiv.data(i) += opts.bwdfn(finput.data(i), foutput.data(i), fderiv.data(i));
		  	  i += 1;
		  	}
		  	inputDeriv <-- finderiv;		    
		  }
		  inplaceNoConnectReleaseDeriv()
		  backwardtime += toc - start;
  }
  
  override def clear = {
    clearMats;
    finput = null;
    foutput = null;
    finderiv = null;
    fderiv = null;
  }
  
  override def toString = {
    "efn@"+Integer.toHexString(hashCode % 0x10000).toString
  }
}

trait EfnNodeOpts extends NodeOpts { 
  var fwdfn:(Float)=> Float = null;
  var bwdfn:(Float,Float,Float)=> Float = null;
}

class EfnNode extends Node with EfnNodeOpts {
  	def copyTo(opts:EfnNode):EfnNode = {
			super.copyTo(opts);
			opts.fwdfn = fwdfn;
			opts.bwdfn = bwdfn;
			opts;
	}

	override def clone:EfnNode = {copyTo(new EfnNode).asInstanceOf[EfnNode];}

  override def create(net:Net):EfnLayer = {EfnLayer(net, this);}
  
  override def toString = {
    "efn@"+Integer.toHexString(hashCode % 0x10000).toString
  }
}

object EfnLayer {  
  
  def apply(net:Net) = new EfnLayer(net, new EfnNode);
  
  def apply(net:Net, opts:EfnNode) = new EfnLayer(net, opts);
}