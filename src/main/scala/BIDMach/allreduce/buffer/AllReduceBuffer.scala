package BIDMach.allreduce.buffer

abstract class AllReduceBuffer(dataSize: Int,
                               peerSize: Int,
                               maxLag: Int,
                               maxChunkSize: Int) {

  type Buffer = Array[Array[Float]]
  val numChunks = getNumChunk(dataSize)
  var temporalBuffer: Array[Buffer] = {
    Array.fill(maxLag) {
      initializePeerBuffer()
    }
  }

  private def initializePeerBuffer(): Buffer = {
    Array.fill(peerSize) {
      Array.fill(dataSize)(0)
    }
  }

  protected val countFilled: Array[Array[Int]] = Array.ofDim[Int](maxLag, numChunks)

  def store(data: Array[Float], round: Int, srcId: Int, chunkId: Int) = {
    val array = temporalBuffer(timeIdx(round))(srcId)
    System.arraycopy(
      data, 0,
      array, chunkId * maxChunkSize,
      data.size)
    countFilled(timeIdx(round))(chunkId) += 1
  }

  protected def timeIdx(round: Int) = {
    round % maxLag
  }

  def up(round: Int): Unit = {
    temporalBuffer(timeIdx(round)) = initializePeerBuffer()
    countFilled(timeIdx(round)) = Array.fill(numChunks)(0);
  }

  protected def getNumChunk(size: Int) = {
    math.ceil(1f * size / maxChunkSize).toInt
  }
}
