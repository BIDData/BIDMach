package BIDMach.allreduce.buffer

case class ScatteredDataBuffer(dataSize: Int,
                               peerSize: Int,
                               maxLag: Int,
                               reducingThreshold: Float,
                               maxChunkSize: Int) extends AllReduceBuffer(dataSize, peerSize, maxLag, maxChunkSize) {

  private val minRequired: Int = (reducingThreshold * peerSize).toInt

  def reachReducingThreshold(row: Int, chunkId: Int): Boolean = {
    countFilled(timeIdx(row))(chunkId) == minRequired
  }


  def count(row: Int, chunkId: Int): Int = {
    countFilled(timeIdx(row))(chunkId)
  }

  def reduce(row : Int, chunkId: Int) : (Array[Float], Int) = {
    val countVal = count(row, chunkId)
    val (unreduced, unreducedChunkSize) = get(row, chunkId)
    val reduced = Array.fill[Float](unreducedChunkSize)(0)
    for (i <- 0 until peerSize) {
      for (j <- 0 until unreducedChunkSize) {
        reduced(j) += unreduced(i)(j)
      }
    }
    (reduced, countVal)
  }

  private def get(row: Int, chunkId: Int): (Buffer, Int) = {
    val endPos = math.min(dataSize, (chunkId + 1) * maxChunkSize)
    val length = endPos - chunkId * maxChunkSize
    val outputSize = temporalBuffer(row).length
    val output: Array[Array[Float]] = new Array[Array[Float]](outputSize)
    for (i <- 0 until outputSize) {
      output(i) = temporalBuffer(timeIdx(row))(i).slice(chunkId * maxChunkSize, endPos)
    }
    (output, length)
  }

}

object ScatteredDataBuffer {
  def empty = {
    ScatteredDataBuffer(0, 0, 0, 0f, 1024)
  }
}