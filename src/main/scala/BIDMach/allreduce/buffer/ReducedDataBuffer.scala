package BIDMach.allreduce.buffer

import java.util

case class ReducedDataBuffer(maxBlockSize: Int,
                             minBlockSize: Int,
                             totalDataSize: Int,
                             peerSize: Int,
                             maxLag: Int,
                             completionThreshold: Float,
                             maxChunkSize: Int) extends AllReduceBuffer(maxBlockSize, peerSize, maxLag, maxChunkSize) {

  val minChunkRequired: Int = {
    val minNumChunks = getNumChunk(minBlockSize)
    val totalChunks = numChunks * (peerSize - 1) + minNumChunks
    (completionThreshold * totalChunks).toInt
  }

  private val countReduceFilled: Array[Array[Int]] = Array.ofDim[Int](maxLag, peerSize * numChunks)

  def store(data: Array[Float], round: Int, srcId: Int, chunkId: Int, count: Int) = {
    super.store(data, round, srcId, chunkId)
    countReduceFilled(timeIdx(round))(srcId * numChunks + chunkId) = count
  }

  def getWithCounts(round: Int, dataOutput: Array[Float], countOutput: Array[Int]) = {
    val output = temporalBuffer(timeIdx(round))
    val countOverPeerChunks = countReduceFilled(timeIdx(round))

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

  override def up(round: Int): Unit = {
    super.up(round)
    util.Arrays.fill(countReduceFilled(timeIdx(round)), 0, peerSize * numChunks, 0)
  }

  def reachCompletionThreshold(round: Int): Boolean = {
    var chunksCompleteReduce = 0
    for (i <- 0 until countFilled(timeIdx(round)).length) {
      chunksCompleteReduce += countFilled(timeIdx(round))(i);
    }
    chunksCompleteReduce == minChunkRequired
  }
}

object ReducedDataBuffer {
  def empty = {
    ReducedDataBuffer(0, 0, 0, 0, 0, 0f, 1024)
  }
}