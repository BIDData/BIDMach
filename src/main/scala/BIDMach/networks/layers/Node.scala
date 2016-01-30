package BIDMach.networks.layers

import BIDMat.{Mat,SBMat,CMat,DMat,FMat,IMat,LMat,HMat,GMat,GDMat,GIMat,GLMat,GSMat,GSDMat,SMat,SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import BIDMach.datasources._
import BIDMach.updaters._
import BIDMach.mixins._
import BIDMach.models._
import BIDMach.networks.layers._
import BIDMach._
import edu.berkeley.bid.CPUMACH
import edu.berkeley.bid.CUMACH
import scala.util.hashing.MurmurHash3;
import java.util.HashMap;
import BIDMach.networks._


@SerialVersionUID(100L)
trait NodeOpts extends BIDMat.Opts {
  var name = "";  
  
  def copyOpts(opts:NodeOpts):NodeOpts = {
    opts.name = name;
		opts;
  }
}

class Node extends NodeTerm(null, 0) with NodeOpts {
	val inputs:Array[NodeTerm] = Array(null);
  var myLayer:Layer = null;
  var myGhost:Node = null;
  var parent:Node = null;
  var outputNumbers:Array[Int] = null;
  
  override def node = this;
  
  def copyTo(opts:Node):Node = {
    copyOpts(opts);
    opts.inputs(0) = inputs(0);
    myGhost = opts;
    opts;
  }
  
  override def toString = {
    "node@"+(hashCode % 0x10000).toString
  }

  override def clone:Node = {
		copyTo(new Node).asInstanceOf[Node];
  } 
  
  def apply(i:Int) = new NodeTerm(this, i);
  
  def create(net:Net):Layer = {null}
}


class NodeTerm(val _node:Node, val term:Int) extends Serializable {  
  
  def node = _node;
  
  def + (a:NodeTerm) = {val n=this; new AddNode{inputs(0)=n; inputs(1)=a}};

  def *@ (a:NodeTerm) = {val n=this; new MulNode{inputs(0)=n; inputs(1)=a;}};
    
  def ∘ (a:NodeTerm) = {val n=this; new MulNode{inputs(0)=n; inputs(1)=a;}};
        
  def on (a:NodeTerm) = {val n=this; new StackNode{inputs(0)=n; inputs(1)=a;}};
}

object Node {
  
  def copy(a:NodeTerm) = new CopyNode{inputs(0) = a;}

  def copy = new CopyNode
  
  def dropout(a:NodeTerm, frac:Float) = new DropoutNode{inputs(0) = a; frac = frac}
  
  def exp(a:NodeTerm) = new ExpNode{inputs(0) = a;};
  
  def GLM(a:NodeTerm)(implicit opts:GLMNodeOpts) = new GLMNode{inputs(0) = a; links = opts.links};
  
  def input(a:NodeTerm) = new InputNode{inputs(0) = a;};
  
  def input = new InputNode
  
  def linear(a:NodeTerm)(name:String="", outdim:Int=0, hasBias:Boolean=true, aopts:ADAGrad.Opts=null) = {
    val odim = outdim;
    val hBias = hasBias;
    val aaopts = aopts;
    val mname = name;
    new LinNode{inputs(0)=a; modelName = mname; outdim=odim; hasBias=hBias; aopts=aaopts};
  }
  
  def linear_(a:NodeTerm)(implicit opts:LinNodeOpts) = {
    val n = new LinNode{inputs(0) = a;}
    opts.copyOpts(n);
    n
  }
  
  def ln(a:NodeTerm) = new LnNode{inputs(0) = a};
  
  def negsamp(a:NodeTerm)(name:String="", outdim:Int=0, hasBias:Boolean=true, aopts:ADAGrad.Opts=null, nsamps:Int=100, expt:Float=0.5f, scoreType:Int=0, doCorrect:Boolean=true) = {
    val odim = outdim;
    val hBias = hasBias;
    val aaopts = aopts;
    val nnsamps = nsamps;
    val eexpt = expt;
    val dcr = doCorrect;
    val sct = scoreType;
    val mname = name;
    new NegsampOutputNode{inputs(0)=a; modelName=mname; outdim=odim; hasBias=hBias; aopts=aaopts; nsamps=nnsamps; expt=eexpt; scoreType=sct; docorrect=dcr};
  }
    
  def negsamp_(a:NodeTerm)(implicit opts:NegsampOutputNodeOpts) =     {
    val n = new NegsampOutputNode{inputs(0) = a}
    opts.copyOpts(n);
    n
  }
  
  def norm(a:NodeTerm)(implicit opts:NormNodeOpts) =  {
    val n = new NormNode{inputs(0) = a;}
    opts.copyOpts(n);
    n
  }
  
  def oneHot(a:NodeTerm) = new OnehotNode{inputs(0) = a};
  
  def rect(a:NodeTerm) = new RectNode{inputs(0) = a};
  
  def sigmoid(a:NodeTerm) = new SigmoidNode{inputs(0) = a};
  
  def σ(a:NodeTerm) = new SigmoidNode{inputs(0) = a};

  def softmax(a:NodeTerm) = new SoftmaxNode{inputs(0) = a};
  
  def softmaxout(a:NodeTerm)(scoreTyp:Int=0) =  new SoftmaxOutputNode{inputs(0) = a; scoreType=scoreTyp}
  
  def softplus(a:NodeTerm) = new SoftplusNode{inputs(0) = a};
  
  def splithoriz(a:NodeTerm, np:Int) = new SplitHorizNode{inputs(0) = a; nparts = np};
  
  def splitvert(a:NodeTerm, np:Int) = new SplitVertNode{inputs(0) = a; nparts = np};
  
  def tanh(a:NodeTerm) = new TanhNode{inputs(0) = a};
  
  def lstm(h:NodeTerm, c:NodeTerm, i:NodeTerm, m:String)(opts:LSTMNodeOpts) = {
    val n = new LSTMNode;
    opts.copyOpts(n);
    n.modelName = m;
    n.constructGraph;
    n.inputs(0) = h;
    n.inputs(1) = c;
    n.inputs(2) = i;
    n
  }
  
  implicit def NodeToNodeMat(n:Node):NodeMat = NodeMat.elem(n);

}