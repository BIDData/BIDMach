package BIDMach.networks.layers

import BIDMach.networks.Net;
import BIDMat.Mat
import BIDMat.IMat
import BIDMat.DenseMat
import scala.collection.mutable.HashMap

case class LayerMat(override val nrows:Int, override val ncols:Int, override val data:Array[Layer]) extends DenseMat[Layer](nrows, ncols, data) {	
	
	override def t:LayerMat = LayerMat(gt(null))
	
	override def mytype = "LayerMat"
	
	def horzcat(b: LayerMat) = LayerMat(ghorzcat(b))
	
	def vertcat(b: LayerMat) = LayerMat(gvertcat(b))
	
	def find3:(IMat, IMat, LayerMat) = { val vv = gfind3 ; (IMat(vv._1), IMat(vv._2), LayerMat(vv._3)) }
	
	override def apply(a:IMat):LayerMat = LayerMat(gapply(a))
	
	override def apply(a:IMat, b:IMat):LayerMat = LayerMat(gapply(a, b))	
	
	override def apply(a:Int, b:IMat):LayerMat = LayerMat(gapply(a, b))	
		
	override def apply(a:IMat, b:Int):LayerMat = LayerMat(gapply(a, b))	
		  
  override def apply(a:Mat, b:Mat):LayerMat = LayerMat(gapply(a.asInstanceOf[IMat], b.asInstanceOf[IMat]))
  
  override def apply(a:Mat, b:Int):LayerMat = LayerMat(gapply(a.asInstanceOf[IMat], b))
  
  override def apply(a:Int, b:Mat):LayerMat = LayerMat(gapply(a, b.asInstanceOf[IMat]))
  
  
  def update(i:Int, b:Layer):Layer = _update(i, b)
  
  def update(i:Int, j:Int, b:Layer):Layer = _update(i, j, b)
  
  
  def update(iv:IMat, b:LayerMat):LayerMat = LayerMat(_update(iv, b))
  
  def update(iv:IMat, jv:IMat, b:LayerMat):LayerMat = LayerMat(_update(iv, jv, b))

  def update(iv:IMat, j:Int, b:LayerMat):LayerMat = LayerMat(_update(iv, IMat.ielem(j), b))

  def update(i:Int, jv:IMat, b:LayerMat):LayerMat = LayerMat(_update(IMat.ielem(i), jv, b))
  
//  override def update(inds:IMat, b:Int):Mat = LayerMat(_update(inds, b.toFloat))
  
//  override def update(inds:IMat, b:Float):Mat = LayerMat(_update(inds, b))
  
  override def update(iv:IMat, b:Mat):LayerMat = LayerMat(_update(iv, b.asInstanceOf[LayerMat]))
  
  override def update(iv:IMat, jv:IMat, b:Mat):LayerMat = LayerMat(_update(iv, jv, b.asInstanceOf[LayerMat]))

  override def update(iv:IMat, j:Int, b:Mat):LayerMat = LayerMat(_update(iv, IMat.ielem(j), b.asInstanceOf[LayerMat]))

  override def update(i:Int, jv:IMat, b:Mat):LayerMat = LayerMat(_update(IMat.ielem(i), jv, b.asInstanceOf[LayerMat]))
   
  override def update(iv:Mat, b:Mat):LayerMat = LayerMat(_update(iv.asInstanceOf[IMat], b.asInstanceOf[LayerMat]))
  
  override def update(iv:Mat, jv:Mat, b:Mat):LayerMat = LayerMat(_update(iv.asInstanceOf[IMat], jv.asInstanceOf[IMat], b.asInstanceOf[LayerMat]))

  override def update(iv:Mat, j:Int, b:Mat):LayerMat = LayerMat(_update(iv.asInstanceOf[IMat], IMat.ielem(j), b.asInstanceOf[LayerMat]))

  override def update(i:Int, jv:Mat, b:Mat):LayerMat = LayerMat(_update(IMat.ielem(i), jv.asInstanceOf[IMat], b.asInstanceOf[LayerMat]))
  
  
  
  def update(iv:Mat, b:Layer):LayerMat = LayerMat(_update(iv.asInstanceOf[IMat], b))
  
  def update(iv:Mat, jv:Mat, b:Layer):LayerMat = LayerMat(_update(iv.asInstanceOf[IMat], jv.asInstanceOf[IMat], b))

  def update(iv:Mat, j:Int, b:Layer):LayerMat = LayerMat(_update(iv.asInstanceOf[IMat], IMat.ielem(j), b))

  def update(i:Int, jv:Mat, b:Layer):LayerMat = LayerMat(_update(IMat.ielem(i), jv.asInstanceOf[IMat], b))
  
	def ccMatOp(b: LayerMat, f:(Layer, Layer) => Layer, old:LayerMat) = LayerMat(ggMatOp(b, f, old))
	
	def ccMatOpScalar(b: Layer, f:(Layer, Layer) => Layer, old:LayerMat) = LayerMat(ggMatOpScalar(b, f, old))
	
	def ccReduceOp(n:Int, f1:(Layer) => Layer, f2:(Layer, Layer) => Layer, old:LayerMat) = LayerMat(ggReduceOp(n, f1, f2, old))
  
  var layerMap:HashMap[Layer,Int] = null;
	
   def rebuildMap = {
    layerMap = new HashMap[Layer,Int]();
    for (i <- 0 until data.length) {
      layerMap.put(data(i), i);
    }
  }
  
  def alphaCoords(layerTerm:LayerTerm) = {
    if (layerTerm == null) {
      "null"
    } else {
    	val layer = layerTerm.layer;
    	val term = layerTerm.term;
    	if (layerMap == null) {
    		rebuildMap;
    	}
    	if (layerMap.contains(layer)) {
    		val i = layerMap(layer);
    		if (data(i) != layer) rebuildMap;
    		val coli = i / nrows;
    		val rowi = i - coli * nrows;
    		val v:Int = 'A';
    		val coli0 = coli % 26;
    		val ch0 = Character.toChars(v + coli0)(0).toString;
    		val ch = if (coli < 26) {
    			ch0;
    		} else {
    			val ch1 = Character.toChars(v + coli0/26)(0).toString;
    			ch1 + ch0;
    		} 
    		val ostr = ch + rowi.toString;  
    		if (term == 0) {
    			ostr;
    		} else {
    			ostr + "[" + term.toString + "]";
    		}
      } else {
        "<==="
      }
    }
  }
  
  override def printOne(i:Int):String = {
    val v = data(i)
    if (v != null) {
      val ostring = v.inputs.map(alphaCoords(_)).reduce(_+","+_);
      v.toString() + "(" + ostring +")";
    }
    else  
      ""
  }

		
	def \ (b: LayerMat) = horzcat(b);
	def \ (b: Layer) = horzcat(LayerMat.elem(b))
	def on (b: LayerMat) = vertcat(b)
	def on (b: Layer) = vertcat(LayerMat.elem(b))
	
	def link(b:LayerMat):Unit = {
	  for (i <- 0 until math.min(nrows, b.nrows)) {
	    val lleft = apply(i, ncols-1);
	    val lright = b(i, 0);
	    (lleft, lright) match {
	      case (a:LSTMLayer, b:LSTMLayer) => {
	        b.setInput(0, a(0));
	        b.setInput(1, a(1));
	      }
	      case _ => {}
	    }
	  }
	}
	
	def forward(col1:Int, col2:Int, debug:Int) = {
	  for (i <- col1 to col2) {
	    for (j <- 0 until nrows) {
	    	if (debug > 0) {
	    		println("  forward (%d,%d) %s" format (j, i, apply(j, i).getClass))
	    	}
	    	apply(j, i).forward;
	    }
	  }
	}
	
	def backward(col1:Int, col2:Int, debug:Int, ipass:Int, ipos:Long) = {
		for (i <- col2 to col2 by -1) {
			for (j <- (nrows-1) to 0 by -1) {
				if (debug > 0) {
					println("  backward (%d,%d) %s" format (j, i, apply(j, i).getClass))
				}
				apply(j, i).backward(ipass, ipos);
			}
		}
	}
}

object LayerMat {
  
    def apply(nr:Int, nc:Int):LayerMat = new LayerMat(nr, nc, new Array[Layer](nr*nc))

    def apply(a:DenseMat[Layer]):LayerMat = new LayerMat(a.nrows, a.ncols, a.data) 
    
    def apply(a:List[Layer]) = new LayerMat(1, a.length, a.toArray)
    
    def apply(n:NodeMat, net:Net):LayerMat = {
      val nr = n.nrows;
      val nc = n.ncols;
      val mat = new LayerMat(nr, nc, new Array[Layer](nr*nc));
      for (i <- 0 until nc) {
        for (j <- 0 until nr) {
          if (n(j, i) != null) {
          	mat(j, i) = n(j, i).create(net);
          	n(j, i).myLayer = mat(j, i);
          }
        }
      }
      for (i <- 0 until nc) {
        for (j <- 0 until nr) {
          if (n(j, i) != null) {
            val inputs = n(j, i).inputs;
            for (k <- 0 until inputs.length) {
              val layer = inputs(k).node.myLayer;
              val layerTerm = if (inputs(k).term != 0) {
                new LayerTerm(layer, inputs(k).term)
              } else {
                layer;
              }
              mat(j, i).setInput(k, layerTerm);
            }
          }
        }
      }
      mat;
    }
    
    def elem(x:Layer) = {
    	val out = LayerMat(1,1)
    	out.data(0) = x
    	out
	}

}






