package BIDMach.allreduce.buffer


abstract class AllReduceBuffer(dataSize: Int,
                               peerSize: Int,
                               maxLag: Int,
                               maxChunkSize: Int) {

  type Buffer = Array[Array[Float]]
  val numChunks = getNumChunk(dataSize)
  val temporalBuffer: Array[Buffer] = {
    val init = new Array[Buffer](maxLag)
    for (i <- 0 until maxLag) {
      init(i) = new Array[Array[Float]](peerSize)
      for (j <- 0 until peerSize) {
        init(i)(j) = new Array[Float](dataSize)
      }
    }
    init
  }

  protected val countFilled: Array[Array[Int]] = Array.ofDim[Int](maxLag, numChunks)

  protected def store(data: Array[Float], round: Int, srcId: Int, chunkId: Int) = {

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

  protected def getNumChunk(size: Int) = {
    math.ceil(1f * size / maxChunkSize).toInt
  }
}
