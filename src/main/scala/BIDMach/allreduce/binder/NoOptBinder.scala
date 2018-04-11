package BIDMach.allreduce.binder
import BIDMach.allreduce.binder.AllreduceBinder.{DataSink, DataSource}

class NoOptBinder(dataSize: Int, printFrequency: Int = 10) extends AllreduceBinder {


  val random = new scala.util.Random(100)
  val totalInputSample = 8

  lazy val randomFloats = {
    val nestedArray = new Array[Array[Float]](totalInputSample)
    for (i <- 0 until totalInputSample) {
      nestedArray(i) = Array.range(0, dataSize).toList.map(_ => random.nextFloat()).toArray
    }
    nestedArray
  }


  override def dataSource: DataSource = { inputRequest =>
    if (inputRequest.iteration % printFrequency == 0) {
      println(s"--NoOptBinder: dump model data at ${inputRequest.iteration}--")
    }
    AllReduceInput(randomFloats(inputRequest.iteration % totalInputSample))
  }

  override def dataSink: DataSink = { output =>
    if (output.iteration % printFrequency == 0) {
      println(s"--NoOptBinder: reduced done data at ${output.iteration}--")
    }
  }

  override def totalDataSize: Int = dataSize
}
