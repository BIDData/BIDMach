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

@SerialVersionUID(100L)
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

@SerialVersionUID(100L)
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
    	val indims = irow(nfeats, nheads, nbatch, nwords);
    	val dims1 = irow(nfeats, nheads, nbatch * nwords);
    	val dims2 = irow(nfeats, nheads * nbatch, nwords);
    	val dims3 = irow(nwords, nheads * nbatch * nwords);
    	val dims4 = irow(nwords, nheads * nbatch, nwords);
    	val dims5 = irow(nfeats * nheads, nbatch * nwords);
      
      val proj1dims = irow(nfeats, nheads, nfeats);
      val proj2dims = irow(nfeats * nheads, nfeats * nheads);
        
    		
    	val v1   = copy;
    	val k1   = copy; 
    	val q1   = copy;
    	val c1   = constant(row(1f/ math.sqrt(dimk).toFloat));
      
      val mv   = variable(proj1dims)();
      val mk   = variable(proj1dims)();
      val mq   = variable(proj1dims)();
      val mout = variable(proj2dims)();

    	val pv   = mv *-* v1
      val pk   = mk *-* k1
      val pq   = mq *-* q1
      
      val v2   = reshape(pv)(dims2);
      val k2   = reshape(pk)(dims2);
      val q2   = reshape(pq)(dims2);
    	val prods  = k2 ^*-* q2;
      
      val scaled = prods *@ c1;        // nwords, nheads * nbatch, nwords
      val sc2    = reshape(scaled)(dims3)
    	val sm     = softmax(sc2);       // softmax over first dimension (keys)      
      val sm2    = reshape(sm)(dims4)
      
    	val sdp    = k2 *-* sm;
      val shaped = reshape(sdp)(dims5);
    	val mapped = mout * sdp;
    	val out    = reshape(mapped)(dims1)

    	grid =  (v1   \   mv   \    pv    \   v2    \  scaled  \     sdp   on
               k1   \   mk   \    pk    \   k2    \     sc2  \  shaped   on
               q1   \   mq   \    pq    \   q2    \      sm  \  mapped   on
               c1   \  mout  \   null   \  prods  \     sm2  \     out   );
    	
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

@SerialVersionUID(100L)
object MultiHeadAttnLayer {    
  
  def apply(net:Net) = new MultiHeadAttnLayer(net, new MultiHeadAttnNode);
  
  def apply(net:Net, opts:MultiHeadAttnNode) = {
    val x = new MultiHeadAttnLayer(net, opts);
    x.construct;
    x;
  }
}
