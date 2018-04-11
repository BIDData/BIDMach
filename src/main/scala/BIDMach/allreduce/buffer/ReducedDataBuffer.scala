package BIDMach.allreduce.buffer

import java.util

case class ReducedDataBuffer(maxBlockSize: Int,
                             minBlockSize: Int,
                             totalDataSize: Int,
                             peerSize: Int,
                             completionThreshold: Float,
                             maxChunkSize: Int) extends AllReduceBuffer(maxBlockSize, peerSize, maxChunkSize) {

  var numChunkReceived: Int = 0

  val minChunkRequired: Int = {
    val minNumChunks = getNumChunk(minBlockSize)
    val totalChunks = numChunks * (peerSize - 1) + minNumChunks
    (completionThreshold * totalChunks).toInt
  }

  private val peerPreReduceDataCount: Array[Array[Int]] = Array.ofDim[Int](peerSize, numChunks)

  def store(data: Array[Float], srcId: Int, chunkId: Int, count: Int) = {
    super.store(data, srcId, chunkId)
    numChunkReceived += 1
    peerPreReduceDataCount(srcId)(chunkId) = count
  }

  def getWithCounts(dataOutput: Array[Float], countOutput: Array[Int]) = {

    var transferred = 0
    var countTransferred = 0

    for (peerId <- 0 until peerSize) {
      val blockFromPeer = peerBuffer(peerId)
      val blockSize = Math.min(totalDataSize - transferred, blockFromPeer.size)
      System.arraycopy(blockFromPeer, 0, dataOutput, transferred, blockSize)

      for (chunkId <- 0 until numChunks) {
        val countChunkSize = {
          val countSize = Math.min(maxChunkSize, maxBlockSize - maxChunkSize * chunkId)
          Math.min(totalDataSize - countTransferred, countSize)
        }
        // duplicate count from chunk to element level
        util.Arrays.fill(countOutput, countTransferred, countTransferred + countChunkSize, peerPreReduceDataCount(peerId)(chunkId))
        countTransferred += countChunkSize
      }
      transferred += blockSize
    }

    (dataOutput, countOutput)
  }

  def prepareNewRound(): Unit = {
    // clear peer data/count buffers
    for (i <- 0 until peerSize) {
      util.Arrays.fill(peerBuffer(i), 0)
      util.Arrays.fill(peerPreReduceDataCount(i), 0)
    }
    numChunkReceived = 0
  }

  def reachCompletionThreshold(): Boolean = {
    numChunkReceived == minChunkRequired
  }
}

object ReducedDataBuffer {
  def empty = {
    ReducedDataBuffer(0, 0, 0, 0, 0f, 0)
  }
}