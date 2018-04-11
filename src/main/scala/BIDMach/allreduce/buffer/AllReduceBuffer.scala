package BIDMach.allreduce.buffer


abstract class AllReduceBuffer(dataSize: Int,
                               peerSize: Int,
                               maxChunkSize: Int) {

  type Buffer = Array[Array[Float]]

  val peerBuffer: Buffer = Array.ofDim(peerSize, dataSize)

  val numChunks = getNumChunk(dataSize)

  protected def store(data: Array[Float], srcId: Int, chunkId: Int) = {

    val array = peerBuffer(srcId)
    System.arraycopy(
      data, 0,
      array, chunkId * maxChunkSize,
      data.size)
  }

  protected def getNumChunk(size: Int) = {
    math.ceil(1f * size / maxChunkSize).toInt
  }
}
