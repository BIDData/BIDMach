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
trait NodeOpts extends Serializable {
  val inputs:Array[NodeOpts] = Array(null);
  val inputTerminals:Array[Int] = Array(0);
  var myLayer:Layer = null;
  var myGhost:NodeOpts = null;
  var parent:Node = null;
  var outputNumbers:Array[Int] = null;
  var name = "";  
  
  def copyTo(opts:NodeOpts):NodeOpts = {
		opts.inputs(0) = inputs(0);
		opts.inputTerminals(0) = inputTerminals(0);
		myGhost = opts;
		opts;
  }
  
  def copyOpts(opts:NodeOpts):NodeOpts = {
		opts;
  }
}

class Node extends NodeTerm(null, 0) with NodeOpts {
  
  override def node = this;

  override def clone:Node = {
		copyTo(new Node).asInstanceOf[Node];
  } 
  
  def apply(i:Int) = new NodeTerm(this, i);
  
  def create(net:Net):Layer = {null}

}

class NodeTerm(val _node:Node, val term:Int) extends Serializable {  
  
  def node = _node;
  
  def + (a:NodeTerm) = {val n=node; val t=term; new AddNode{inputs(0)=n; inputTerminals(0)=t; inputs(1)=a.node; inputTerminals(1)=a.term}};

  def *@ (a:NodeTerm) = {val n=node; val t=term; new MulNode{inputs(0)=n; inputTerminals(0)=t; inputs(1)=a.node; inputTerminals(1)=a.term}};
    
  def ∘ (a:NodeTerm) = {val n=node; val t=term; new MulNode{inputs(0)=n; inputTerminals(0)=t; inputs(1)=a.node; inputTerminals(1)=a.term}};
        
  def on (a:NodeTerm) = {val n=node; val t=term; new StackNode{inputs(0)=n; inputTerminals(0)=t; inputs(1)=a.node; inputTerminals(1)=a.term}};
}

object Node {
  
  def _getNode(a:NodeTerm):Node = if (a != null) a.node else null;
  
  def _getTerm(a:NodeTerm):Int = if (a != null) a.term else 0;
  
  def copy(a:NodeTerm) = new CopyNode{inputs(0) = _getNode(a); inputTerminals(0) = _getTerm(a);}

  def copy = new CopyNode
  
  def dropout(a:NodeTerm, frac:Float) = new DropoutNode{inputs(0) = _getNode(a); inputTerminals(0) = _getTerm(a); frac = frac}
  
  def exp(a:NodeTerm) = new ExpNode{inputs(0) = _getNode(a); inputTerminals(0) = _getTerm(a)};
  
  def GLM(a:NodeTerm)(implicit opts:GLMNodeOpts) = new GLMNode{inputs(0) = _getNode(a); inputTerminals(0) = _getTerm(a); links = opts.links};
  
  def input(a:NodeTerm) = new InputNode{inputs(0) = _getNode(a); inputTerminals(0) = _getTerm(a)};
  
  def input = new InputNode
  
  def linear(a:NodeTerm)(name:String="", outdim:Int=0, hasBias:Boolean=true, aopts:ADAGrad.Opts=null) = {
    val odim = outdim;
    val hBias = hasBias;
    val aaopts = aopts
    new LinNode{inputs(0)=_getNode(a); inputTerminals(0)=_getTerm(a); modelName = name; outdim=odim; hasBias=hBias; aopts=aaopts};
  }
  
  def linear_(a:NodeTerm)(implicit opts:LinNodeOpts) = {
    val n = new LinNode{inputs(0) = _getNode(a); inputTerminals(0) = _getTerm(a);}
    opts.copyOpts(n);
    n
  }
  
  def ln(a:NodeTerm) = new LnNode{inputs(0) = _getNode(a); inputTerminals(0) = _getTerm(a)};
  
  def negsamp_(a:NodeTerm)(implicit opts:NegsampOutputNodeOpts) =     {
    val n = new NegsampOutputNode{inputs(0) = _getNode(a); inputTerminals(0) = _getTerm(a);}
    opts.copyOpts(n);
    n
  }
  
  def negsamp(a:NodeTerm)(name:String="", outdim:Int=0, hasBias:Boolean=true, aopts:ADAGrad.Opts=null, nsamps:Int=100, expt:Float=0.5f, scoreType:Int=0, doCorrect:Boolean=true) = {
    val odim = outdim;
    val hBias = hasBias;
    val aaopts = aopts;
    val nnsamps = nsamps;
    val eexpt = expt;
    val dcr = doCorrect;
    val sct = scoreType;
    new NegsampOutputNode{inputs(0)=_getNode(a); inputTerminals(0)=_getTerm(a); modelName=name; outdim=odim; hasBias=hBias; aopts=aaopts; nsamps=nnsamps; expt=eexpt; scoreType=sct; docorrect=dcr};
  }
  
  def norm(a:NodeTerm)(implicit opts:NormNodeOpts) =  {
    val n = new NormNode{inputs(0) = _getNode(a); inputTerminals(0) = _getTerm(a);}
    opts.copyOpts(n);
    n
  }
  
  def oneHot(a:NodeTerm) = new OnehotNode{inputs(0) = _getNode(a); inputTerminals(0) = _getTerm(a)};
  
  def rect(a:NodeTerm) = new RectNode{inputs(0) = _getNode(a); inputTerminals(0) = _getTerm(a)};
  
  def sigmoid(a:NodeTerm) = new SigmoidNode{inputs(0) = _getNode(a); inputTerminals(0) = _getTerm(a)};
  
  def σ(a:NodeTerm) = new SigmoidNode{inputs(0) = _getNode(a); inputTerminals(0) = _getTerm(a)};

  def softmax(a:NodeTerm) = new SoftmaxNode{inputs(0) = _getNode(a); inputTerminals(0) = _getTerm(a)};
  
  def softmaxout(a:NodeTerm)(scoreTyp:Int=0) =  new SoftmaxOutputNode{inputs(0) = _getNode(a); inputTerminals(0) = _getTerm(a); scoreType=scoreTyp}
  
  def softplus(a:NodeTerm) = new SoftplusNode{inputs(0) = _getNode(a); inputTerminals(0) = _getTerm(a)};
  
  def splithoriz(a:NodeTerm, np:Int) = new SplitHorizNode{inputs(0) = _getNode(a); inputTerminals(0) = _getTerm(a); nparts = np};
  
  def splitvert(a:NodeTerm, np:Int) = new SplitVertNode{inputs(0) = _getNode(a); inputTerminals(0) = _getTerm(a); nparts = np};
  
  def tanh(a:NodeTerm) = new TanhNode{inputs(0) = _getNode(a); inputTerminals(0) = _getTerm(a)};
  
  def lstm(h:NodeTerm, c:NodeTerm, i:NodeTerm)(opts:LSTMNodeOpts) = {
    val n = new LSTMNode;
    opts.copyOpts(n);
    n.constructGraph;
    n.inputs(0) = _getNode(h);
    n.inputs(1) = _getNode(c);
    n.inputs(2) = _getNode(i);
    n.inputTerminals(0) = _getTerm(h);
    n.inputTerminals(1) = _getTerm(c);
    n.inputTerminals(2) = _getTerm(i);
    n
  }
  
  implicit def NodeToNodeMat(n:Node):NodeMat = NodeMat.elem(n);

}