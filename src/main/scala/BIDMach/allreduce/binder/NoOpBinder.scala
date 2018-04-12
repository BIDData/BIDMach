package BIDMach.allreduce.binder
import BIDMach.allreduce.binder.AllreduceBinder.{DataSink, DataSource}

class NoOpBinder(dataSize: Int, printFrequency: Int = 10) extends AllreduceBinder {


  val random = new scala.util.Random(100)
  val totalInputSample = 4

  lazy val randomFloats = {
    val nestedArray: Array[Array[Float]] = Array.ofDim(totalInputSample, dataSize)
    for (i <- 0 until totalInputSample) {
      for (j <- 0 until dataSize)
      nestedArray(i)(j) = random.nextFloat()
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
