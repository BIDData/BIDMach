package BIDMach.allreduce.buffer

import org.scalatest.{Matchers, WordSpec}

import scala.util.Random

class ScatteredDataBufferSpec extends WordSpec with Matchers {


  "Scattered buffer behavior" should {

    val blockSize = 5
    val peerSize = 4
    val maxLag = 4
    val reducingThreshold = 0.75f
    val maxChunkSize = 3
    val rowAtTest = 1
    val chunkAtTest = 0

    val buffer = ScatteredDataBuffer(blockSize, peerSize, reducingThreshold, maxChunkSize)
    val numChunks = buffer.numChunks
    val expectedCriticalPeerSize = 3


    "initialize buffers" in {

      buffer.temporalBuffer.length shouldEqual peerSize
      buffer.temporalBuffer(0).length shouldEqual blockSize

    }

    "throw exception when data to store at the end exceeds expected size" in {

      val lastChunkId = numChunks - 1
      intercept[ArrayIndexOutOfBoundsException] {
        val toStore: Array[Float] = randomFloatArray(maxChunkSize)
        buffer.store(toStore, 0, lastChunkId)
      }
      val excess = numChunks * maxChunkSize - blockSize
      val toStore = randomFloatArray(maxChunkSize - excess)
      buffer.store(toStore, 0, lastChunkId)
    }

    "reach reducing threshold" in {

      val reachingThresholdChunkId = 0
      val reachThreshold = List(false, false, true)
      for (i <- 0 until expectedCriticalPeerSize) {
        val toStore = randomFloatArray(maxChunkSize)
        buffer.store(toStore, srcId = i, reachingThresholdChunkId)
        buffer.reachReducingThreshold(reachingThresholdChunkId) shouldBe reachThreshold(i)
      }

    }

    "reduce values with correct count" in {
      val (nonEmptyReduced, counts) = buffer.reduce(chunkAtTest)
      counts shouldEqual expectedCriticalPeerSize
      nonEmptyReduced.sum shouldNot equal(0)
    }

    "be unable to store older round and have buffer at chunk cleared after preparation for new round" in {

      val newRoundOfSameMod = rowAtTest + maxLag

      buffer.prepareNewRound()

      val (reduced, count) = buffer.reduce(chunkAtTest)
      count shouldEqual 0
      reduced.toList shouldEqual Array.fill(reduced.size)(0)
    }

  }

  "Scattered buffer summation reduction" should {

    val blockSize = 2
    val maxChunkSize = 3
    val peerSize = 2
    val maxLag = 2
    val rowAtTest = 0
    val chunkAtTest = 0
    val reducingThreshold = 1f

    val buffer = ScatteredDataBuffer(blockSize, peerSize, reducingThreshold, maxChunkSize)

    "sum from all peers at one row" in {

      for (i <- 0 until peerSize) {
        val arrayToStore = (0 until blockSize).map(_ => i.toFloat).toArray
        buffer.store(data = arrayToStore, srcId = i, chunkId = chunkAtTest)
        val (_, count) = buffer.reduce(chunkId = chunkAtTest)
        count shouldEqual i + 1
      }

      val (reducedSum, _) = buffer.reduce(chunkId = chunkAtTest)
      val sequenceSum = (0 until peerSize).sum
      reducedSum.toList shouldEqual (0 until blockSize).map(_ => sequenceSum)

    }

  }

  private def randomFloatArray(maxChunkSize: Int) = {
    Array.range(0, maxChunkSize).toList.map(_ => Random.nextFloat()).toArray
  }
}
