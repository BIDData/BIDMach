package BIDMach.allreduce.buffer

import java.util

case class ScatteredDataBuffer(dataSize: Int,
                               peerSize: Int,
                               maxLag: Int,
                               reducingThreshold: Float,
                               maxChunkSize: Int) extends AllReduceBuffer(dataSize, peerSize, maxLag, maxChunkSize) {

  val minChunkRequired: Int = (reducingThreshold * peerSize).toInt

  private val currentRounds: Array[Array[Int]] = {
    val rounds = new Array[Array[Int]](maxLag)
    for (i <- 0 until maxLag) {
      rounds(i) = new Array[Int](numChunks)
      java.util.Arrays.fill(rounds(i), i)
    }
    rounds
  }

  def compareRoundTo(round: Int, chunkId: Int): Int = {
    currentRounds(timeIdx(round))(chunkId).compareTo(round)
  }

  def getRound(round: Int, chunkId: Int): Int = {
    currentRounds(timeIdx(round))(chunkId)
  }

  def count(round: Int, chunkId: Int): Int = {
    countFilled(timeIdx(round))(chunkId)
  }

  override def store(data: Array[Float], round: Int, srcId: Int, chunkId: Int) = {
    if (compareRoundTo(round, chunkId) > 0) {
      throw new IllegalArgumentException(s"Unable to store data chunk $chunkId from source $srcId, as given round [$round] is less than current round [${currentRounds(timeIdx(round))(chunkId)}]")
    }
    super.store(data, round, srcId, chunkId)
  }


  def reduce(round : Int, chunkId: Int) : (Array[Float], Int) = {

    val chunkStartPos = chunkId * maxChunkSize
    val chunkEndPos = math.min(dataSize, (chunkId + 1) * maxChunkSize)
    val chunkSize = chunkEndPos - chunkStartPos
    val reducedArr = new Array[Float](chunkSize)
    for (i <- 0 until peerSize) {
      val tBuf = temporalBuffer(timeIdx(round))(i);
      var j = 0;
      while (j < chunkSize) {
        reducedArr(j) += tBuf(chunkStartPos + j);
        j += 1;
      }
    }
    (reducedArr, count(round, chunkId))
  }

  def prepareNewRound(round : Int, chunkId: Int) = {

    currentRounds(timeIdx(round))(chunkId) += maxLag

    val chunkStartPos = chunkId * maxChunkSize
    val chunkEndPos = math.min(dataSize, (chunkId + 1) * maxChunkSize)
    val tBuf = temporalBuffer(timeIdx(round))
    for(peerId <- 0 until peerSize) {
      util.Arrays.fill(
        tBuf(peerId),
        chunkStartPos,
        chunkEndPos,
        0
      )
    }
    countFilled(timeIdx(round))(chunkId) = 0
  }

  def reachReducingThreshold(round: Int, chunkId: Int): Boolean = {
    countFilled(timeIdx(round))(chunkId) == minChunkRequired
  }

}

object ScatteredDataBuffer {
  def empty = {
    ScatteredDataBuffer(0, 0, 0, 0f, 0)
  }
}