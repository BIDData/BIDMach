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

class Node extends NodeOpts {
  
  val terminal = 0;

  override def clone:Node = {
		copyTo(new Node).asInstanceOf[Node];
  } 
  
  def apply(i:Int) = new NodeTerm(this, i);
  
  def create(net:Net):Layer = {null}
  
  def + (a:Node) = new AddNode{inputs(0) = this; inputs(1) = a};
  
  def *@ (a:Node) = new MulNode{inputs(0) = this; inputs(1) = a};
  
  def ∘ (a:Node) = new MulNode{inputs(0) = this; inputs(1) = a};
  
  def on (a:Node) = new StackNode{inputs(0) = this; inputs(1) = a};
  
  
  def + (a:NodeTerm) = new AddNode{inputs(0) = this; inputs(1) = a.node; inputTerminals(1) = a.term};
  
  def *@ (a:NodeTerm) = new MulNode{inputs(0) = this; inputs(1) = a.node; inputTerminals(1) = a.term};
  
  def ∘ (a:NodeTerm) = new MulNode{inputs(0) = this; inputs(1) = a.node; inputTerminals(1) = a.term};
  
  def on (a:NodeTerm) = new StackNode{inputs(0) = this; inputs(1) = a.node; inputTerminals(1) = a.term};

}

class NodeTerm(val node:Node, val term:Int) {
  def + (a:Node) = new AddNode{inputs(0) = node; inputTerminals(0) = term; inputs(1) = a};
  
  def *@ (a:Node) = new MulNode{inputs(0) = node; inputTerminals(0) = term; inputs(1) = a};
  
  def ∘ (a:Node) = new MulNode{inputs(0) = node; inputTerminals(0) = term; inputs(1) = a};
  
  def on (a:Node) = new StackNode{inputs(0) = node; inputTerminals(0) = term; inputs(1) = a};
  
  
  def + (a:NodeTerm) = new AddNode{inputs(0) = node; inputTerminals(0) = term; inputs(1) = a.node; inputTerminals(1) = a.term};

  def *@ (a:NodeTerm) = new MulNode{inputs(0) = node; inputTerminals(0) = term; inputs(1) = a.node; inputTerminals(1) = a.term};
    
  def ∘ (a:NodeTerm) = new MulNode{inputs(0) = node; inputTerminals(0) = term; inputs(1) = a.node; inputTerminals(1) = a.term};
        
  def on (a:NodeTerm) = new StackNode{inputs(0) = node; inputTerminals(0) = term; inputs(1) = a.node; inputTerminals(1) = a.term};
}

object Node {
  def copy(a:Node) = new CopyNode{inputs(0) = a}
  
  def dropout(a:Node, frac:Float) = new DropoutNode{inputs(0) = a; frac = frac}
  
  def exp(a:Node) = new ExpNode{inputs(0) = a};
  
  def GLM(a:Node)(implicit opts:GLMNodeOpts) = new GLMNode{inputs(0) = a; links = opts.links};
  
  def input(a:Node) = new InputNode{inputs(0) = a};
  
  def linear(a:Node)(implicit opts:LinNodeOpts) = {
    val n = new LinNode{inputs(0) = a;}
    opts.copyOpts(n);
    n
  }
  
  def linear(a:Node, name:String)(implicit opts:LinNodeOpts) = {
    val n = new LinNode{inputs(0) = a;}
    opts.copyOpts(n);
    n.modelName = name;
    n
  }
  
  def ln(a:Node) = new LnNode{inputs(0) = a};
  
  def negsamp(a:Node)(implicit opts:NegsampOutputNodeOpts) =     {
    val n = new NegsampOutputNode{inputs(0) = a;}
    opts.copyOpts(n);
    n
  }
  
  def norm(a:Node)(implicit opts:NormNodeOpts) =  {
    val n = new NormNode{inputs(0) = a;}
    opts.copyOpts(n);
    n
  }
  
  def oneHot(a:Node) = new OnehotNode{inputs(0) = a};
  
  def rect(a:Node) = new RectNode{inputs(0) = a};
  
  def sigmoid(a:Node) = new SigmoidNode{inputs(0) = a};
  
  def σ(a:Node) = new SigmoidNode{inputs(0) = a};
  
  def softmax(a:Node) = new SoftmaxNode{inputs(0) = a};
  
  def softmaxout(a:Node)(implicit opts:SoftmaxOutputNodeOpts) =  {
    val n = new NormNode{inputs(0) = a;}
    opts.copyOpts(n);
    n
  }
  
  def softplus(a:Node) = new SoftplusNode{inputs(0) = a};
  
  def splithoriz(a:Node, np:Int) = new SplitHorizNode{inputs(0) = a; nparts = np};
  
  def splitvert(a:Node, np:Int) = new SplitVertNode{inputs(0) = a; nparts = np};
  
  def tanh(a:Node) = new TanhNode{inputs(0) = a};
  
  
  
  def copy(a:NodeTerm) = new CopyNode{inputs(0) = a.node; inputTerminals(0) = a.term}
  
  def dropout(a:NodeTerm, frac:Float) = new DropoutNode{inputs(0) = a.node; inputTerminals(0) = a.term; frac = frac}
  
  def exp(a:NodeTerm) = new ExpNode{inputs(0) = a.node; inputTerminals(0) = a.term};
  
  def GLM(a:NodeTerm)(implicit opts:GLMNodeOpts) = new GLMNode{inputs(0) = a.node; inputTerminals(0) = a.term; links = opts.links};
  
  def input(a:NodeTerm) = new InputNode{inputs(0) = a.node; inputTerminals(0) = a.term};
  
  def linear(a:NodeTerm)(implicit opts:LinNodeOpts) = {
    val n = new LinNode{inputs(0) = a.node; inputTerminals(0) = a.term;}
    opts.copyOpts(n);
    n
  }
  
  def linear(a:NodeTerm, name:String)(implicit opts:LinNodeOpts) = {
    val n = new LinNode{inputs(0) = a.node; inputTerminals(0) = a.term;}
    opts.copyOpts(n);
    opts.modelName = name;
    n
  }
  
  def ln(a:NodeTerm) = new LnNode{inputs(0) = a.node; inputTerminals(0) = a.term};
  
  def negsamp(a:NodeTerm)(implicit opts:NegsampOutputNodeOpts) =     {
    val n = new NegsampOutputNode{inputs(0) = a.node; inputTerminals(0) = a.term;}
    opts.copyOpts(n);
    n
  }
  
  def norm(a:NodeTerm)(implicit opts:NormNodeOpts) =  {
    val n = new NormNode{inputs(0) = a.node; inputTerminals(0) = a.term;}
    opts.copyOpts(n);
    n
  }
  
  def oneHot(a:NodeTerm) = new OnehotNode{inputs(0) = a.node; inputTerminals(0) = a.term};
  
  def rect(a:NodeTerm) = new RectNode{inputs(0) = a.node; inputTerminals(0) = a.term};
  
  def sigmoid(a:NodeTerm) = new SigmoidNode{inputs(0) = a.node; inputTerminals(0) = a.term};
  
  def σ(a:NodeTerm) = new SigmoidNode{inputs(0) = a.node; inputTerminals(0) = a.term};

  def softmax(a:NodeTerm) = new SoftmaxNode{inputs(0) = a.node; inputTerminals(0) = a.term};
  
  def softmaxout(a:NodeTerm)(implicit opts:SoftmaxOutputNodeOpts) =  {
    val n = new NormNode{inputs(0) = a.node; inputTerminals(0) = a.term;}
    opts.copyOpts(n);
    n
  }
  
  def softplus(a:NodeTerm) = new SoftplusNode{inputs(0) = a.node; inputTerminals(0) = a.term};
  
  def splithoriz(a:NodeTerm, np:Int) = new SplitHorizNode{inputs(0) = a.node; inputTerminals(0) = a.term; nparts = np};
  
  def splitvert(a:NodeTerm, np:Int) = new SplitVertNode{inputs(0) = a.node; inputTerminals(0) = a.term; nparts = np};
  
  def tanh(a:NodeTerm) = new TanhNode{inputs(0) = a.node; inputTerminals(0) = a.term};
  

}