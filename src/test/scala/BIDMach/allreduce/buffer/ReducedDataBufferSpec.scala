package BIDMach.allreduce.buffer

import org.scalatest.{Matchers, WordSpec}

import scala.util.Random

class ReducedDataBufferSpec extends WordSpec with Matchers {

  "Reduced buffer behavior on evenly sized blocks" should {

    val maxBlockSize = 5
    val minBlockSize = 5
    val peerSize = 3
    val maxLag = 4
    val threshold = 0.7f
    val maxChunkSize = 2

    val totalDataSize = 15
    val rowAtTest = 1

    val outputBuff = new Array[Float](totalDataSize)
    val outputCount = new Array[Int](totalDataSize)

    val buffer = ReducedDataBuffer(maxBlockSize, minBlockSize, totalDataSize, peerSize, maxLag, threshold, maxChunkSize)

    "initialize buffers" in {

      buffer.temporalBuffer.length shouldEqual maxLag
      buffer.temporalBuffer(0).length shouldEqual peerSize
      buffer.temporalBuffer(0)(0).length shouldEqual maxBlockSize

    }

    "have zero counts" in {

      buffer.getWithCounts(rowAtTest, outputBuff, outputCount)
      outputBuff.sum shouldEqual 0
      outputCount.sum shouldEqual 0

    }

    "store first peer first chunk data" in {
      val srcId = 0
      val chunkId = 0
      val toStore: Array[Float] = randomFloatArray(maxChunkSize)
      buffer.store(toStore, rowAtTest, srcId, chunkId, count = peerSize)
      buffer.getWithCounts(rowAtTest, outputBuff, outputCount)
      outputBuff.toList.slice(0, maxChunkSize) shouldEqual toStore.toList

      for (i <- 0 until maxChunkSize) {
        outputCount(i) shouldEqual peerSize
      }

    }

    "store last peer last chunk with smaller size" in {
      val srcId = peerSize - 1
      val chunkId = buffer.numChunks - 1
      intercept[ArrayIndexOutOfBoundsException] {
        val toStore: Array[Float] = randomFloatArray(maxChunkSize)
        buffer.store(toStore, rowAtTest, srcId, chunkId, count = peerSize)
      }

      val lastChunkSize = maxBlockSize - (buffer.numChunks - 1) * maxChunkSize
      val toStore: Array[Float] = randomFloatArray(lastChunkSize)
      buffer.store(toStore, rowAtTest, srcId, chunkId, count = peerSize)

      buffer.getWithCounts(rowAtTest, outputBuff, outputCount)

      outputBuff.toList.slice(totalDataSize - lastChunkSize, totalDataSize) shouldEqual toStore.toList

    }

    "store until reach completion threshold " in {

      buffer.reachCompletionThreshold(rowAtTest) shouldBe false

      // 6 chunks of reduced data required
      // 0.7 * 3 * 3 - (threshold * peer size * num chunks)

      // peer 0, second chunk
      buffer.store(randomFloatArray(maxChunkSize), round = 1, srcId = 0, chunkId = 1, count = peerSize)
      buffer.reachCompletionThreshold(rowAtTest) shouldBe false

      // peer 1, first, second chunk
      buffer.store(randomFloatArray(maxChunkSize), round = 1, srcId = 1, chunkId = 0, count = peerSize)
      buffer.store(randomFloatArray(maxChunkSize), round = 1, srcId = 1, chunkId = 1, count = peerSize)
      buffer.reachCompletionThreshold(rowAtTest) shouldBe false

      // peer 2, second chunk
      buffer.store(randomFloatArray(maxChunkSize), round = 1, srcId = 2, chunkId = 1, count = peerSize)
      buffer.reachCompletionThreshold(rowAtTest) shouldBe true

    }


    "get reduced row" in {
      // peer 0 - missing 3rd chunk
      // peer 1 - missing 3rd chunk
      // peer 2 - missing 1st chunk

      buffer.getWithCounts(rowAtTest, outputBuff, outputCount)

      val missingIndex = List(4, 9, 10, 11)
      for (i <- missingIndex) {
        outputBuff(i) shouldEqual 0
      }


      // count is zero when data is missing
      for (i <- missingIndex) {
        outputCount(i) shouldEqual 0
      }

      val presentIndex = (0 until totalDataSize).filterNot(missingIndex.contains)
      for (i <- presentIndex) {
        outputCount(i) shouldEqual peerSize
      }

    }

  }


  "Reduced buffer behavior on uneven sized blocks" should {

    val maxBlockSize = 6
    val minBlockSize = 4
    val peerSize = 3
    val maxLag = 4
    val threshold = 1
    val maxChunkSize = 2

    val totalDataSize = 16
    val rowAtTest = 1

    val buffer = ReducedDataBuffer(maxBlockSize, minBlockSize, totalDataSize, peerSize, maxLag, threshold, maxChunkSize)

    "store until reach completion threshold " in {

      buffer.reachCompletionThreshold(rowAtTest) shouldBe false

      // 8 chunks

      // peer 0/1
      for (chunkId <- 0 until 3) {
        for (peerId <- 0 until 2) {
          buffer.store(randomFloatArray(maxChunkSize), round = rowAtTest, srcId = peerId, chunkId = chunkId, count = peerSize)
          buffer.reachCompletionThreshold(rowAtTest) shouldBe false
        }
      }

      buffer.store(randomFloatArray(maxChunkSize), round = rowAtTest, srcId = 2, chunkId = 0, count = peerSize)
      buffer.reachCompletionThreshold(rowAtTest) shouldBe false

      buffer.store(randomFloatArray(maxChunkSize), round = rowAtTest, srcId = 2, chunkId = 1, count = peerSize)
      buffer.reachCompletionThreshold(rowAtTest) shouldBe true

    }

  }


  private def randomFloatArray(maxChunkSize: Int) = {
    Array.range(0, maxChunkSize).toList.map(_ => Random.nextFloat()).toArray
  }

}
