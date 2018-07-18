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
 * 
 */

class MultiHeadAttnLayer(override val net:Net, override val opts:MultiHeadAttnNode = new MultiHeadAttnNode) extends CompoundLayer(net, opts) {
	override val _inputs = new Array[LayerTerm](3);
	override val _outputs = new Array[Mat](2);
	override val _derivs = new Array[Mat](2);
  
  override def toString = {
    "mhattn@"+Integer.toHexString(hashCode % 0x10000).toString
  }
}

trait MultiHeadAttnNodeOpts extends CompoundNodeOpts {
    var dimk = 0;
    var dimv = 0;
    var dimmodel = 0;
    var nfeats = 0;
    var nheads = 0;
    var nbatch = 0;
    var nwords = 0;
    var kind = 1;
    var hasBias = false;  
    
   def copyOpts(opts:MultiHeadAttnNodeOpts):MultiHeadAttnNodeOpts = {
  		super.copyOpts(opts);
  		opts.dimv = dimv;
  		opts.dimk = dimk;
  		opts.dimmodel = dimmodel;
  		opts.kind = kind;
  		opts.hasBias = hasBias;
  		opts;
    }
}

class MultiHeadAttnNode extends CompoundNode with MultiHeadAttnNodeOpts {	
  
	  override val inputs:Array[NodeTerm] = Array(null, null, null);
//    override val inputTerminals:Array[Int] = Array(0,0,0);
	  
    def v = apply(0);
    
    def k = apply(1);
    
    def q = apply(2);
    
    def constructGraph = {
      val nodelist = Net.defaultNodeList;
	    Net.defaultNodeList = null;
	    
      kind match {
        case 0 => constructGraph0
        case _ => throw new RuntimeException("MultiHeadAttnLayer type %d not recognized" format kind);
      }
      
      Net.defaultNodeList = nodelist;
    }
  
    // Basic MultiHeadAttn topology with 8 linear layers
	  	  
	  def constructGraph0 = {
    	import BIDMach.networks.layers.Node._
    	val indims = irow(nfeats, nbatch, nwords, nheads);
    	val outdims = irow(nfeats * nheads, nbatch, nwords);
    		
    	val in_v = copy;
    	val in_k = copy; 
    	val in_q = copy;
    	val c1   = constant(row(1f/ math.sqrt(dimk).toFloat));
    	
    	val v_split = reshape(in_v)(indims);
    	val k_split = reshape(in_k)(indims);
    	val q_split = reshape(in_q)(indims);

    	val v_int  = linear(v_split)(prefix+"MultiHeadAttn_v_internal",   outdim=dimmodel, hasBias=hasBias);
    	val k_int  = linear(k_split)(prefix+"MultiHeadAttn_k_internal",   outdim=dimmodel, hasBias=hasBias);   
    	val q_int  = linear(q_split)(prefix+"MultiHeadAttn_q_internal",   outdim=dimmodel, hasBias=hasBias);
    	
    	val prod1  = q_int * k_int;
    	val prod1a = prod1 *@ c1;
    	val sm     = softmax(prod1a);
    	val sdp    = sm * v_split
    	
    	val merged = reshape(sdp)(outdims)
    	

    	val out    = merged

    	grid =  (in_v   on
               in_k   on
               in_q   on
               c1   );
    	
    	val lopts = grid.data;
    	lopts.map((x:Node) => if (x != null) x.parent = this);
    	outputNumbers = Array(lopts.indexOf(out));
	  }
	  
	  override def clone:MultiHeadAttnNode = {
		  copyTo(new MultiHeadAttnNode).asInstanceOf[MultiHeadAttnNode];
	  }

	  override def create(net:Net):MultiHeadAttnLayer = {
		  MultiHeadAttnLayer(net, this);
	  }
    
    override def toString = {
    "mhattn@"+Integer.toHexString(hashCode % 0x10000).toString
    }
	}


  
object MultiHeadAttnNode {   
  
  final val gridTypeNoOutput = 0;
  final val gridTypeSoftmaxOutput = 1;
  final val gridTypeNegsampOutput = 2;
  final val gridTypeSoftmax = 3;
  
  def apply() = {
    val n = new MultiHeadAttnNode;
    n.constructGraph;
    n
  }
  
  def apply(opts:MultiHeadAttnNodeOpts) = {
    val n = new MultiHeadAttnNode;
    opts.copyOpts(n);
    n.constructGraph;
    n
  }
}

object MultiHeadAttnLayer {    
  
  def apply(net:Net) = new MultiHeadAttnLayer(net, new MultiHeadAttnNode);
  
  def apply(net:Net, opts:MultiHeadAttnNode) = {
    val x = new MultiHeadAttnLayer(net, opts);
    x.construct;
    x;
  }
}
