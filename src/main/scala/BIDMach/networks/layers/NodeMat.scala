package BIDMach.networks.layers
import BIDMat.Mat
import BIDMat.IMat
import BIDMat.DenseMat
import scala.collection.mutable.HashMap

case class NodeMat(override val nrows:Int, override val ncols:Int, override val data:Array[Node]) extends DenseMat[Node](nrows, ncols, data) {  
  
  var nodeMap:HashMap[Node,Int] = null
  
  override def t:NodeMat = NodeMat(gt(null))
  
  override def mytype = "NodeMat"
  
  def horzcat(b: NodeMat) = NodeMat(ghorzcat(b))
  
  def vertcat(b: NodeMat) = NodeMat(gvertcat(b))
  
  def find3:(IMat, IMat, NodeMat) = { val vv = gfind3 ; (IMat(vv._1), IMat(vv._2), NodeMat(vv._3)) }
  
  override def apply(a:IMat):NodeMat = NodeMat(gapply(a))
  
  override def apply(a:IMat, b:IMat):NodeMat = NodeMat(gapply(a, b))  
  
  override def apply(a:Int, b:IMat):NodeMat = NodeMat(gapply(a, b)) 
    
  override def apply(a:IMat, b:Int):NodeMat = NodeMat(gapply(a, b)) 
      
  override def apply(a:Mat, b:Mat):NodeMat = NodeMat(gapply(a.asInstanceOf[IMat], b.asInstanceOf[IMat]))
  
  override def apply(a:Mat, b:Int):NodeMat = NodeMat(gapply(a.asInstanceOf[IMat], b))
  
  override def apply(a:Int, b:Mat):NodeMat = NodeMat(gapply(a, b.asInstanceOf[IMat]))
  
  
  def update(i:Int, b:Node):Node = _update(i, b)
  
  def update(i:Int, j:Int, b:Node):Node = _update(i, j, b)
  
  
  def update(iv:IMat, b:NodeMat):NodeMat = NodeMat(_update(iv, b))
  
  def update(iv:IMat, jv:IMat, b:NodeMat):NodeMat = NodeMat(_update(iv, jv, b))

  def update(iv:IMat, j:Int, b:NodeMat):NodeMat = NodeMat(_update(iv, IMat.ielem(j), b))

  def update(i:Int, jv:IMat, b:NodeMat):NodeMat = NodeMat(_update(IMat.ielem(i), jv, b))
  
//  override def update(inds:IMat, b:Int):Mat = NodeMat(_update(inds, b.toFloat))
  
//  override def update(inds:IMat, b:Float):Mat = NodeMat(_update(inds, b))
  
  override def update(iv:IMat, b:Mat):NodeMat = NodeMat(_update(iv, b.asInstanceOf[NodeMat]))
  
  override def update(iv:IMat, jv:IMat, b:Mat):NodeMat = NodeMat(_update(iv, jv, b.asInstanceOf[NodeMat]))

  override def update(iv:IMat, j:Int, b:Mat):NodeMat = NodeMat(_update(iv, IMat.ielem(j), b.asInstanceOf[NodeMat]))

  override def update(i:Int, jv:IMat, b:Mat):NodeMat = NodeMat(_update(IMat.ielem(i), jv, b.asInstanceOf[NodeMat]))
   
  override def update(iv:Mat, b:Mat):NodeMat = NodeMat(_update(iv.asInstanceOf[IMat], b.asInstanceOf[NodeMat]))
  
  override def update(iv:Mat, jv:Mat, b:Mat):NodeMat = NodeMat(_update(iv.asInstanceOf[IMat], jv.asInstanceOf[IMat], b.asInstanceOf[NodeMat]))

  override def update(iv:Mat, j:Int, b:Mat):NodeMat = NodeMat(_update(iv.asInstanceOf[IMat], IMat.ielem(j), b.asInstanceOf[NodeMat]))

  override def update(i:Int, jv:Mat, b:Mat):NodeMat = NodeMat(_update(IMat.ielem(i), jv.asInstanceOf[IMat], b.asInstanceOf[NodeMat]))
  
  def update(iv:Mat, b:Node):NodeMat = NodeMat(_update(iv.asInstanceOf[IMat], b))
  
  def update(iv:Mat, jv:Mat, b:Node):NodeMat = NodeMat(_update(iv.asInstanceOf[IMat], jv.asInstanceOf[IMat], b))

  def update(iv:Mat, j:Int, b:Node):NodeMat = NodeMat(_update(iv.asInstanceOf[IMat], IMat.ielem(j), b))

  def update(i:Int, jv:Mat, b:Node):NodeMat = NodeMat(_update(IMat.ielem(i), jv.asInstanceOf[IMat], b))
  
  def ccMatOp(b: NodeMat, f:(Node, Node) => Node, old:NodeMat) = NodeMat(ggMatOp(b, f, old))
  
  def ccMatOpScalar(b: Node, f:(Node, Node) => Node, old:NodeMat) = NodeMat(ggMatOpScalar(b, f, old))
  
  def ccReduceOp(n:Int, f1:(Node) => Node, f2:(Node, Node) => Node, old:NodeMat) = NodeMat(ggReduceOp(n, f1, f2, old))
  
  def map(f: Node => Layer) = {
    val out = LayerMat(nrows, ncols)
    for (i <- 0 until length) {
      out(i) = f(data(i))
    }
    out
  }
  
  def rebuildMap = {
    nodeMap = new HashMap[Node,Int]()
    for (i <- 0 until data.length) {
      nodeMap(data(i)) = i
    }
  }
  
  def alphaCoords(nodeTerm:NodeTerm) = {
    if (nodeTerm == null) {
      "null"
    } else {
      val node = nodeTerm.node
      val term = nodeTerm.term
      if (nodeMap == null) {
        rebuildMap
      }
      if (nodeMap.contains(node)) {
        val i = nodeMap(node)
        if (data(i) != node) rebuildMap
        val coli = i / nrows
        val rowi = i - coli * nrows
        val v:Int = 'A'
        val coli0 = coli % 26
        val ch0 = Character.toChars(v + coli0)(0).toString
        val ch = if (coli < 26) {
          ch0
        } else {
          val ch1 = Character.toChars(v + coli0/26)(0).toString
          ch1 + ch0
        } 
        val ostr = ch + rowi.toString;  
        if (term == 0) {
          ostr
        } else {
          ostr + "[" + term.toString + "]"
        }
      } else {
        "<==="
      }
    }
  }
  
  override def printOne(i:Int):String = {
    val v = data(i)
    if (v != null) {
      val ostring = v.inputs.map(alphaCoords(_)).reduce(_+","+_)
      v.toString() + "(" + ostring +")"
    }
    else  
      ""
  }
    
  def \ (b: NodeMat) = horzcat(b)
  def \ (b: Node) = horzcat(NodeMat.elem(b))
  def on (b: NodeMat) = vertcat(b)
  def on (b: Node) = vertcat(NodeMat.elem(b))
}

object NodeMat {
  
    def apply(nr:Int, nc:Int):NodeMat = new NodeMat(nr, nc, new Array[Node](nr*nc))

    def apply(a:DenseMat[Node]):NodeMat = new NodeMat(a.nrows, a.ncols, a.data) 
    
    def apply(a:List[Node]) = new NodeMat(1, a.length, a.toArray)
    
    def elem(x:Node) = {
      val out = NodeMat(1,1)
      out.data(0) = x
      out
  }

}






