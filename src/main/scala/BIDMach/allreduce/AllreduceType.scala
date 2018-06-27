package BIDMach.allreduce

object AllreduceType extends Enumeration {
  val Average, Sum = Value
  type AllreduceType = Value
}

trait ScatterReducer {

  def reduce(partialResult: Float, nextElement: Float): Float

  def postProcess(reducedResult: Float, count: Int): Float

}

object AverageReducer extends ScatterReducer {
  override def reduce(partialResult: Float, nextElement: Float): Float = {
    partialResult + nextElement
  }

  override def postProcess(accumulatedResult: Float, count: Int): Float = {
    accumulatedResult / count
  }
}

object SumReducer extends ScatterReducer {
  override def reduce(partialResult: Float, nextElement: Float): Float = {
    partialResult + nextElement
  }

  override def postProcess(accumulatedResult: Float, count: Int): Float = {
    accumulatedResult
  }
}

object NoOpReducer extends ScatterReducer {
  override def reduce(partialResult: Float, nextElement: Float): Float = {
    partialResult
  }

  override def postProcess(accumulatedResult: Float, count: Int): Float = {
    accumulatedResult
  }
}
