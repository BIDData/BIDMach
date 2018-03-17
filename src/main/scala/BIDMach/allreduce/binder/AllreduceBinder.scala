package BIDMach.allreduce.binder

import BIDMach.allreduce.binder.AllreduceBinder.{DataSink, DataSource}

/**
  * Trait to specify source and sink, allowing binding data input/output to the all-reduce process.
  */
trait AllreduceBinder {

  def totalDataSize: Int

  def dataSource: DataSource

  def dataSink: DataSink

}

object AllreduceBinder {

  type DataSink = AllReduceOutput => Unit
  type DataSource = AllReduceInputRequest => AllReduceInput

}

case class AllReduceInputRequest(iteration: Int)

case class AllReduceInput(data: Array[Float])

case class AllReduceOutput(data: Array[Float], count: Array[Int], iteration: Int)

