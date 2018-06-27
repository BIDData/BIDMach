package BIDMach.allreduce.binder

import BIDMach.allreduce.binder.AllreduceBinder.{DataSink, DataSource}


class AssertCorrectnessBinder(dataSize: Int, checkpoint: Int) extends AllreduceBinder {

  val random = new scala.util.Random(100)
  val totalInputSample = 8

  lazy val randomFloats = {
    val nestedArray = new Array[Array[Float]](totalInputSample)
    for (i <- 0 until totalInputSample) {
      nestedArray(i) = Array.range(0, dataSize).toList.map(_ => random.nextFloat()).toArray
    }
    nestedArray
  }

  private def ~=(x: Double, y: Double, precision: Double = 1e-5) = {
    if ((x - y).abs < precision) true else false
  }

  override def dataSource: DataSource = r => {
    AllReduceInput(randomFloats(r.iteration % totalInputSample))
  }

  override def dataSink: DataSink = r => {

    if (r.iteration % checkpoint == 0) {
      val inputUsed = randomFloats(r.iteration % totalInputSample)
      println(s"\n----Asserting #${r.iteration} output...")
      for (i <- 0 until dataSize) {
        val meanActual = r.data(i)
        val expected = inputUsed(i)
        assert(~=(expected, meanActual), s"Expected [$expected], but actual [$meanActual] at pos $i for iteraton #${r.iteration}")
      }
      println("OK: Means match the expected value!")
    }

  }

  override def totalDataSize: Int = dataSize
}


