package BIDMach.allreduce.buffer

import java.util

case class ReducedDataBuffer(dataSize: Int,
                             peerSize: Int,
                             maxLag: Int,
                             completionThreshold: Float,
                             maxChunkSize: Int) extends AllReduceBuffer(dataSize, peerSize, maxLag, maxChunkSize) {

  private val minChunksRequired: Int = (completionThreshold * peerSize * numChunks).toInt
  private val countReduceFilled: Array[Array[Int]] = Array.ofDim[Int](maxLag, peerSize * numChunks)

  def store(data: Array[Float], row: Int, srcId: Int, chunkId: Int, count: Int) = {
    super.store(data, row, srcId, chunkId)
    countReduceFilled(timeIdx(row))(srcId * numChunks + chunkId) = count
  }

  def getWithCounts(row: Int, totalSize: Int): (Array[Float], Array[Int]) = {
    val output = temporalBuffer(timeIdx(row))
    val countOverPeerChunks = countReduceFilled(timeIdx(row))

    val dataOutput = Array.fill[Float](totalSize)(0.0f)
    val countOutput = Array.fill[Int](totalSize)(0)
    var transferred = 0
    var countTransferred = 0

    for (i <- 0 until peerSize) {
      val blockFromPeer = output(i)
      val blockSize = Math.min(totalSize - transferred, blockFromPeer.size)
      System.arraycopy(blockFromPeer, 0, dataOutput, transferred, blockSize)

      for (j <- 0 until numChunks) {
        val countChunkSize = {
          val countSize = Math.min(maxChunkSize, dataSize - maxChunkSize * j)
          Math.min(totalSize - countTransferred, countSize)
        }
        // duplicate count from chunk to element level
        util.Arrays.fill(countOutput, countTransferred, countTransferred + countChunkSize, countOverPeerChunks(i * numChunks + j))
        countTransferred += countChunkSize
      }
      transferred += blockSize
    }

    (dataOutput, countOutput)
  }

  override def up(): Unit = {
    super.up()
    countReduceFilled(timeIdx(maxLag - 1)) = Array.fill(peerSize * numChunks)(0)
  }

  def reachCompletionThreshold(row: Int): Boolean = {
    var chunksCompleteReduce = 0
    for (i <- 0 until countFilled(row).length) {
      chunksCompleteReduce += countFilled(timeIdx(row))(i);
    }
    chunksCompleteReduce == minChunksRequired
  }
}

object ReducedDataBuffer {
  def empty = {
    ReducedDataBuffer(0, 0, 0, 0f, 1024)
  }
}