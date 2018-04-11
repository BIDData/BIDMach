package BIDMach.allreduce.buffer

import BIDMach.allreduce.{AverageReducer, SumReducer}
import org.scalatest.{Matchers, WordSpec}

import scala.util.Random

class ScatteredDataBufferSpec extends WordSpec with Matchers {


  "Scattered buffer" should {

    val dataSize = 5
    val peerSize = 4
    val reducingThreshold = 0.75f
    val maxChunkSize = 3
    val chunkAtTest = 0

    val buffer = ScatteredDataBuffer(dataSize, peerSize, reducingThreshold, maxChunkSize, SumReducer)
    val numChunks = buffer.numChunks
    val expectedCriticalPeerSize = 3


    "initialize buffers with static values" in {

      buffer.peerBuffer.length shouldEqual peerSize
      buffer.peerBuffer(0).length shouldEqual dataSize

      buffer.numChunks shouldEqual 2
    }


    "throw exception when data to store at the end exceeds expected size" in {

      val lastChunkId = numChunks - 1
      intercept[ArrayIndexOutOfBoundsException] {
        val toStore: Array[Float] = randomFloatArray(maxChunkSize)
        buffer.store(toStore, 0, lastChunkId)
      }
      val excess = numChunks * maxChunkSize - dataSize
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

      buffer.prepareNewRound()

      val (reduced, count) = buffer.reduce(chunkAtTest)
      count shouldEqual 0
      reduced.toList shouldEqual Array.fill(reduced.size)(0)
    }

  }

  "Scattered buffer summation reduction" should {

    val datSize = 10
    val maxChunkSize = 3
    val peerSize = 4
    val chunkAtTest = 0
    val reducingThreshold = 1f

    val buffer = ScatteredDataBuffer(datSize, peerSize, reducingThreshold, maxChunkSize, SumReducer)

    "sum from all peers at one row" in {

      for (i <- 0 until peerSize) {
        val arrayToStore = (0 until maxChunkSize).map(_ => i.toFloat).toArray
        buffer.store(data = arrayToStore, srcId = i, chunkId = chunkAtTest)
        val (_, count) = buffer.reduce(chunkId = chunkAtTest)
        count shouldEqual i + 1
      }

      val (reducedSum, _) = buffer.reduce(chunkId = chunkAtTest)
      val sequenceSum = (0 until peerSize).sum
      reducedSum.toList shouldEqual (0 until maxChunkSize).map(_ => sequenceSum)

    }

  }

  "Scattered buffer average reduction" should {

    val blockSize = 100
    val maxChunkSize = 17
    val peerSize = 20
    val lastChunkId = 5
    val lastChunkSize = 15
    val reducingThreshold = 1f

    val buffer = ScatteredDataBuffer(blockSize, peerSize, reducingThreshold, maxChunkSize, AverageReducer)

    "average from all peers at one row" in {

      val arraysByPeer: List[Array[Float]] = (0 until peerSize).map(_ => randomFloatArray(lastChunkSize)).toList

      for (i <- 0 until peerSize) {
        val arrayToStore = arraysByPeer(i)
        buffer.store(data = arrayToStore, srcId = i, chunkId = lastChunkId)
        val (_, count) = buffer.reduce(chunkId = lastChunkId)
        count shouldEqual i + 1
      }

      val expectedAvg = new Array[Float](lastChunkSize)

      for (i <- 0 until peerSize) {
        for (j <- 0 until lastChunkSize) {
          expectedAvg(j) += arraysByPeer(i)(j)
          if (i == peerSize - 1) {
            expectedAvg(j) /= peerSize
          }
        }
      }

      val (averaged, _) = buffer.reduce(chunkId = lastChunkId)
      averaged.toList shouldEqual expectedAvg.toList

    }

  }


  private def randomFloatArray(maxChunkSize: Int): Array[Float] = {
    Array.range(0, maxChunkSize).toList.map(_ => Random.nextFloat()).toArray
  }


}
