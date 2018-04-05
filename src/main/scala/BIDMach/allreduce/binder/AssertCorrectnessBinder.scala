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
      var zeroCountNum = 0
      var totalCount = 0
      for (i <- 0 until dataSize) {
        val meanActual = r.data(i)
        val count = r.count(i)
        totalCount += count
        if (count == 0) {
          zeroCountNum += 1
        } else {
          val expected = inputUsed(i)
          assert(~=(expected, meanActual), s"Expected [$expected], but actual [$meanActual] at pos $i for iteraton #${r.iteration}")
        }
      }
      val nonZeroCountElementNum = dataSize - zeroCountNum
      println("OK: Mean of non-zero elements match the expected input!")
      println(f"Element with non-zero counts: ${nonZeroCountElementNum / dataSize.toFloat}%.2f ($nonZeroCountElementNum/$dataSize)")
      println(f"Average count value: ${totalCount / nonZeroCountElementNum.toFloat}%2.2f ($totalCount/$nonZeroCountElementNum)")
    }

  }

  override def totalDataSize: Int = dataSize
}


