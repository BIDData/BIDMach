

package BIDMach.allreduce

object AllreduceType extends Enumeration {
  val Average, Sum = Value
  type AllreduceType = Value
}

trait ScatterReducer {

  def reduce(partialResult: Float, nextElement: Float): Float

  def postProcess(reducedResult: Float, count: Int): Float

  def reduceVec(a: Array[Float], aoff:Int, b: Array[Float], n:Int): Unit

  def postProcessVec(a: Array[Float], count:Int, n:Int): Unit

}

object SumReducer extends ScatterReducer {
  def reduce(partialResult: Float, nextElement: Float): Float = {
    partialResult + nextElement
  }

  def postProcess(accumulatedResult: Float, count: Int): Float = {
    accumulatedResult
  }

  def reduceVec(a: Array[Float], aoff:Int, b: Array[Float], n:Int): Unit = { 
    var ai = aoff;
    var i = 0;
    while (i < n) { 
      b(i) += a(ai);
      ai += 1;
      i += 1;
    }
  }

  def postProcessVec(a: Array[Float], count:Int, n:Int):Unit = { 
  }
}

object AverageReducer extends ScatterReducer {
  def reduce(partialResult: Float, nextElement: Float): Float = {
    partialResult + nextElement
  }

  def postProcess(accumulatedResult: Float, count: Int): Float = {
    accumulatedResult
  }

  def reduceVec(a: Array[Float], aoff:Int, b: Array[Float], n:Int): Unit = { 
    var ai = aoff;
    var i = 0;
    while (i < n) { 
      b(i) += a(ai);
      ai += 1;
      i += 1;
    }
  }

  def postProcessVec(a: Array[Float], count:Int, n:Int):Unit = { 
    var i = 0;
    while (i < n) { 
      a(i) = a(i) / count
      i += 1
    }
  }
}


object NoOpReducer extends ScatterReducer {
  def reduce(partialResult: Float, nextElement: Float): Float = {
    partialResult
  }

  def postProcess(accumulatedResult: Float, count: Int): Float = {
    accumulatedResult
  }

  def reduceVec(a: Array[Float], aoff:Int, b: Array[Float], n:Int): Unit = { }

  def postProcessVec(a: Array[Float], count:Int, n:Int): Unit = { }

}
