package BIDMach.allreduce.buffer

import java.util

case class ReducedDataBuffer(maxBlockSize: Int,
                             minBlockSize: Int,
                             totalDataSize: Int,
                             peerSize: Int,
                             completionThreshold: Float,
                             maxChunkSize: Int) extends AllReduceBuffer(maxBlockSize, peerSize, maxChunkSize) {

  val minChunkRequired: Int = {
    val minNumChunks = getNumChunk(minBlockSize)
    val totalChunks = numChunks * (peerSize - 1) + minNumChunks
    (completionThreshold * totalChunks).toInt
  }

  private val countReduceFilled: Array[Int] = Array.ofDim[Int](peerSize * numChunks)

  def store(data: Array[Float], srcId: Int, chunkId: Int, count: Int) = {
    super.store(data, srcId, chunkId)
    countReduceFilled(srcId * numChunks + chunkId) = count
  }

  def getWithCounts(dataOutput: Array[Float], countOutput: Array[Int]) = {


    val output = temporalBuffer
    val countOverPeerChunks = countReduceFilled

    var transferred = 0
    var countTransferred = 0

    for (i <- 0 until peerSize) {
      val blockFromPeer = output(i)
      val blockSize = Math.min(totalDataSize - transferred, blockFromPeer.size)
      System.arraycopy(blockFromPeer, 0, dataOutput, transferred, blockSize)

      for (j <- 0 until numChunks) {
        val countChunkSize = {
          val countSize = Math.min(maxChunkSize, maxBlockSize - maxChunkSize * j)
          Math.min(totalDataSize - countTransferred, countSize)
        }
        // duplicate count from chunk to element level
        util.Arrays.fill(countOutput, countTransferred, countTransferred + countChunkSize, countOverPeerChunks(i * numChunks + j))
        countTransferred += countChunkSize
      }
      transferred += blockSize
    }

    (dataOutput, countOutput)
  }

  def prepareNewRound(): Unit = {
    // clear peer buffers
    val tmpBuff = temporalBuffer
    for (i <- 0 until peerSize) {
      util.Arrays.fill(tmpBuff(i), 0)
    }
    // clear two kinds of count
    util.Arrays.fill(countFilled, 0, numChunks, 0)
    util.Arrays.fill(countReduceFilled, 0, peerSize * numChunks, 0)
  }

  def reachCompletionThreshold(): Boolean = {
    var chunksCompleteReduce = 0
    for (i <- 0 until countFilled.length) {
      chunksCompleteReduce += countFilled(i);
    }
    chunksCompleteReduce == minChunkRequired
  }
}

object ReducedDataBuffer {
  def empty = {
    ReducedDataBuffer(0, 0, 0, 0, 0f, 0)
  }
}