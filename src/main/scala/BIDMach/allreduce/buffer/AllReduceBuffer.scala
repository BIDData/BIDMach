package BIDMach.allreduce.buffer

abstract class AllReduceBuffer(dataSize: Int,
                               peerSize: Int,
                               maxLag: Int,
                               maxChunkSize: Int) {

  type Buffer = Array[Array[Float]]
  var temporalOffset = 0
  val numChunks = math.ceil(1f * dataSize / maxChunkSize).toInt
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

  def store(data: Array[Float], row: Int, srcId: Int, chunkId: Int) = {
    val array = temporalBuffer(timeIdx(row))(srcId)
    System.arraycopy(
      data, 0,
      array, chunkId * maxChunkSize,
      data.size)
    countFilled(timeIdx(row))(chunkId) += 1
  }

  protected def timeIdx(row: Int) = {
    (row + temporalOffset) % maxLag
  }

  def up(): Unit = {
    temporalOffset = (temporalOffset + 1) % maxLag
    temporalBuffer(timeIdx(maxLag - 1)) = initializePeerBuffer()
    countFilled(timeIdx(maxLag - 1)) = Array.fill(numChunks)(0);
  }

}
