package BIDMach.networks.layers

import BIDMat.{Mat,SBMat,CMat,CSMat,DMat,FMat,IMat,LMat,HMat,GMat,GDMat,GIMat,GLMat,GSMat,GSDMat,SMat,SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import BIDMach.datasources._
import BIDMach.updaters._
import BIDMach.mixins._
import BIDMach.models._
import BIDMach.networks._
import BIDMach._
import scala.util.hashing.MurmurHash3;
import scala.collection.mutable.HashMap;

/**
 * LRN within channel unit
 */

class LRNwithinLayer(override val net:Net, override val opts:LRNwithinNode = new LRNwithinNode) extends CompoundLayer(net, opts) {
  
  override def toString = {
    "LRNwithin@"+Integer.toHexString(hashCode % 0x10000).toString
  }
}

trait LRNwithinNodeOpts extends CompoundNodeOpts {
    var dim = 5;
    var alpha = 1f;
    var beta = 0.5f
    
   def copyOpts(opts:LRNwithinNodeOpts):LRNwithinNodeOpts = {
  		super.copyOpts(opts);
  		opts.dim = dim;
  		opts.alpha = alpha;
  		opts.beta = beta;
  		opts;
    }
}

class LRNwithinNode extends CompoundNode with LRNwithinNodeOpts {	
	  	  
	  def constructGraph = {
    	import BIDMach.networks.layers.Node._
    	import jcuda.jcudnn.cudnnPoolingMode._
    	val odim = dim;
    		
    	val in =       copy;
    	val xalpha =   constant(alpha);
    	val xbeta =    constant(beta);
    	val xone =     constant(1f);

    	val prod1 =    in *@ in;
    	val pool1 =    pool(prod1)(h=dim, w=dim, pad=dim/2, poolingMode=CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING);
    	val scale1 =   pool1 *@ xalpha;   	
    	val sum1 =     scale1 + xone;
    	
      val log1 =     ln(sum1);
      val plog1 =    log1 *@ xbeta;
      val div1 =     exp(plog1);
      val out =      in / div1;

    	grid = (in     \ xalpha  \ xbeta    \ xone    on
    	        prod1  \ pool1   \ scale1   \ sum1    on
    	        log1   \ plog1   \ div1     \ out).t
    	
    	val lopts = grid.data;
    	lopts.map((x:Node) => if (x != null) x.parent = this);
    	outputNumbers = Array(lopts.indexOf(out));
	  }
	 
	  override def clone:LRNwithinNode = {
		  copyTo(new LRNwithinNode).asInstanceOf[LRNwithinNode];
	  }

	  override def create(net:Net):LRNwithinLayer = {
		  LRNwithinLayer(net, this);
	  }
    
    override def toString = {
      "LRNwithin@"+Integer.toHexString(hashCode % 0x10000).toString
    }

	}


  
object LRNwithinNode {   
  
  def apply() = {
    val n = new LRNwithinNode;
    n.constructGraph;
    n
  }
  
  def apply(opts:LRNwithinNodeOpts) = {
    val n = new LRNwithinNode;
    opts.copyOpts(n);
    n.constructGraph;
    n
  }
}

object LRNwithinLayer {    
  
  def apply(net:Net) = new LRNwithinLayer(net, new LRNwithinNode);
  
  def apply(net:Net, opts:LRNwithinNode) = {
    val x = new LRNwithinLayer(net, opts);
    x.construct;
    x;
  }
}
