

package BIDMach.allreduce

object AllreduceType extends Enumeration {
  val Average, Sum = Value
  type AllreduceType = Value
}

trait ScatterReducer {

  def reduce(partialResult: Float, nextElement: Float): Float

  def postProcess(reducedResult: Float, count: Int): Float

  def reduceVec(a: Array[Float], aoff:Int, b: Array[Float], boff:Int, c:Array[Float], coff:Int, n:Int): Unit

  def postProcessVec(a: Array[Float], aoff:Int, count:Int, c:Array[Float], coff:Int, n:Int): Unit

}

object AverageReducer extends ScatterReducer {
  override def reduce(partialResult: Float, nextElement: Float): Float = {
    partialResult + nextElement
  }

  override def postProcess(accumulatedResult: Float, count: Int): Float = {
    accumulatedResult / count
  }

  override def reduceVec(a: Array[Float], aoff:Int, b: Array[Float], boff:Int, c:Array[Float], coff:Int, n:Int): Unit = { 
    var ai = aoff;
    var bi = boff;
    var ci = coff;
    var i = 0;
    while (i < n) { 
      c(ci) = a(ai) + b(bi);
      ai += 1;
      bi += 1;
      ci += 1;
      i += 1;
    }
  }

  override def postProcessVec(a: Array[Float], aoff:Int, count:Int, c:Array[Float], coff:Int, n:Int):Unit = { 
    var ai = aoff;
    var ci = coff;
    var i = 0;
    while (i < n) { 
      c(ci) = a(ai) / count
      ai += 1;
      ci += 1;
      i += 1
    }
  }
}

object SumReducer extends ScatterReducer {
  override def reduce(partialResult: Float, nextElement: Float): Float = {
    partialResult + nextElement
  }

  override def postProcess(accumulatedResult: Float, count: Int): Float = {
    accumulatedResult
  }

  override def reduceVec(a: Array[Float], aoff:Int, b: Array[Float], boff:Int, c:Array[Float], coff:Int, n:Int):Unit = { 
    var ai = aoff;
    var bi = boff;
    var ci = coff;
    var i = 0;
    while (i < n) { 
      c(ci) = a(ai) + b(bi);
      ai += 1;
      bi += 1;
      ci += 1;
      i += 1;
    }
  }

  override def postProcessVec(a: Array[Float], aoff:Int, count:Int, c:Array[Float], coff:Int, n:Int):Unit = { 
    var ai = aoff;
    var ci = coff;
    var i = 0;
    while (i < n) { 
      c(ci) = a(ai);
      ai += 1;
      ci += 1;
      i += 1
    }
  }
}

object NoOpReducer extends ScatterReducer {
  override def reduce(partialResult: Float, nextElement: Float): Float = {
    partialResult
  }

  override def postProcess(accumulatedResult: Float, count: Int): Float = {
    accumulatedResult
  }

  override def reduceVec(a: Array[Float], aoff:Int, b: Array[Float], boff:Int, c:Array[Float], coff:Int, n:Int):Unit = { 
  }

  override def postProcessVec(a: Array[Float], aoff:Int, count:Int, c:Array[Float], coff:Int, n:Int):Unit = { 
  }
}
