package BIDMach.networks.layers

@SerialVersionUID(100L)
class NodeSet(val nnodes:Int, val nodes:Array[Node]) extends Serializable {
  
  def this(nnodes:Int) = this(nnodes, new Array[Node](nnodes));
  
  def this(nodes:Array[Node]) = this(nodes.length, nodes);
  
  def apply(i:Int):Node = nodes(i);
  
  def update(i:Int, lopts:Node) = {nodes(i) = lopts; this}
  
  def size = nnodes;

  def length = nnodes;
  
  override def clone = copyTo(new NodeSet(nnodes));
  
  def copyTo(lopts:NodeSet):NodeSet = {
    for (i <- 0 until nnodes) {
      lopts.nodes(i) = nodes(i).clone;
      nodes(i).myGhost = lopts.nodes(i);
    }
    for (i <- 0 until nnodes) {
      for (j <- 0 until nodes(i).inputs.length) {
      	if (nodes(i).inputs(j) != null) lopts.nodes(i).inputs(j) = nodes(i).inputs(j).node.myGhost;
      }
    }
    lopts;
  }
}
