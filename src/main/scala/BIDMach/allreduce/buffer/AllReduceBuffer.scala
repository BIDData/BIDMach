package BIDMach.allreduce.buffer


abstract class AllReduceBuffer(dataSize: Int,
                               peerSize: Int,
                               maxChunkSize: Int) {

  type Buffer = Array[Array[Float]]
  val numChunks = getNumChunk(dataSize)
  // 3D array: maxLag * peerSize * blockSize
  val temporalBuffer: Buffer = {

    val init = new Array[Array[Float]](peerSize)
    for (j <- 0 until peerSize) {
      init(j) = new Array[Float](dataSize)
    }
    init
  }

  // record number of chunks received for each round
  // 2D array: maxLag * numChunksOfEachBlock ?? (should be maxLag * numBlocks?)
  // start reducing block when receive enough chunks for that block of a round
  protected val countFilled: Array[Int] = Array.ofDim[Int](numChunks)

  protected def store(data: Array[Float], srcId: Int, chunkId: Int) = {

    val array = temporalBuffer(srcId)
    System.arraycopy(
      data, 0,
      array, chunkId * maxChunkSize,
      data.size)
    countFilled(chunkId) += 1
  }

  protected def getNumChunk(size: Int) = {
    math.ceil(1f * size / maxChunkSize).toInt
  }
}
