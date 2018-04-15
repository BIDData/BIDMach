package BIDMach.allreduce.buffer

import org.scalatest.{Matchers, WordSpec}

import scala.util.Random

class ReducedDataBufferSpec extends WordSpec with Matchers {

  "Reduced buffer behavior on evenly sized blocks" should {

    val maxBlockSize = 5
    val minBlockSize = 5
    val peerSize = 3
    val threshold = 0.7f
    val maxChunkSize = 2

    val totalDataSize = 15

    val outputBuff = new Array[Float](totalDataSize)
    val emptyBuff = new Array[Float](totalDataSize)

    val buffer = ReducedDataBuffer(maxBlockSize, minBlockSize, totalDataSize, peerSize, threshold, maxChunkSize)

    "initialize buffers" in {

      buffer.peerBuffer.length shouldEqual peerSize
      buffer.peerBuffer(0).length shouldEqual maxBlockSize

    }

    "have zero counts" in {

      buffer.getReducedData(outputBuff, emptyBuff)
      outputBuff.sum shouldEqual 0

    }

    "store first peer first chunk data" in {

      val srcId = 0
      val chunkId = 0
      val toStore: Array[Float] = randomFloatArray(maxChunkSize)
      buffer.store(toStore, srcId, chunkId, count = peerSize)
      buffer.getReducedData(outputBuff, emptyBuff)
      outputBuff.toList.slice(0, maxChunkSize) shouldEqual toStore.toList

    }

    "store last peer last chunk with smaller size" in {
      val srcId = peerSize - 1
      val chunkId = buffer.numChunks - 1
      intercept[ArrayIndexOutOfBoundsException] {
        val toStore: Array[Float] = randomFloatArray(maxChunkSize)
        buffer.store(toStore, srcId, chunkId, count = peerSize)
      }

      val lastChunkSize = maxBlockSize - (buffer.numChunks - 1) * maxChunkSize
      val toStore: Array[Float] = randomFloatArray(lastChunkSize)
      buffer.store(toStore, srcId, chunkId, count = peerSize)

      buffer.getReducedData(outputBuff, emptyBuff)

      outputBuff.toList.slice(totalDataSize - lastChunkSize, totalDataSize) shouldEqual toStore.toList

    }

    "store until reach completion threshold " in {

      buffer.reachCompletionThreshold() shouldBe false

      // 6 chunks of reduced data required
      // 0.7 * 3 * 3 - (threshold * peer size * num chunks)

      // peer 0, second chunk
      buffer.store(randomFloatArray(maxChunkSize), srcId = 0, chunkId = 1, count = peerSize)
      buffer.reachCompletionThreshold() shouldBe false

      // peer 1, first, second chunk
      buffer.store(randomFloatArray(maxChunkSize), srcId = 1, chunkId = 0, count = peerSize)
      buffer.store(randomFloatArray(maxChunkSize), srcId = 1, chunkId = 1, count = peerSize)
      buffer.reachCompletionThreshold() shouldBe false

      // peer 2, second chunk
      buffer.store(randomFloatArray(maxChunkSize), srcId = 2, chunkId = 1, count = peerSize)
      buffer.reachCompletionThreshold() shouldBe true

    }


    "get reduced row maintaining missing pieces" in {
      // peer 0 - missing 3rd chunk
      // peer 1 - missing 3rd chunk
      // peer 2 - missing 1st chunk

      buffer.getReducedData(outputBuff, emptyBuff)

      val missingIndex = List(4, 9, 10, 11)
      for (i <- 0 until totalDataSize) {
        if (missingIndex.contains(i)) outputBuff(i) shouldEqual 0
        else outputBuff(i) should not equal 0
      }

    }

    "get reduced row and fill missing with backup" in {
      // peer 0 - missing 3rd chunk
      // peer 1 - missing 3rd chunk
      // peer 2 - missing 1st chunk
      buffer.getReducedData(outputBuff, emptyBuff)
      val outputWithZeros = Array.ofDim[Float](totalDataSize)

      System.arraycopy(outputBuff, 0, outputWithZeros, 0, totalDataSize)

      val backup = randomFloatArray(totalDataSize)

      buffer.getReducedData(outputBuff, backup)

      val missingIndex = List(4, 9, 10, 11)
      for (i <- 0 until totalDataSize) {
        if (missingIndex.contains(i)) outputBuff(i) shouldEqual backup(i)
        else outputBuff(i) shouldEqual outputWithZeros(i)
      }

    }

    "reset counts and buffer after preparation for new round" in {


      buffer.prepareNewRound()

      buffer.getReducedData(outputBuff, emptyBuff)

      buffer.numChunkReceived shouldEqual 0

      buffer.reachCompletionThreshold() shouldBe false
      outputBuff.toList shouldEqual Array.fill(totalDataSize)(0)

    }



  }


  "Reduced buffer behavior on uneven sized blocks" should {

    val maxBlockSize = 6
    val minBlockSize = 4
    val peerSize = 3
    val threshold = 1
    val maxChunkSize = 2

    val totalDataSize = 16

    val buffer = ReducedDataBuffer(maxBlockSize, minBlockSize, totalDataSize, peerSize, threshold, maxChunkSize)

    "store until reach completion threshold " in {

      buffer.reachCompletionThreshold() shouldBe false

      // 8 chunks

      // peer 0/1
      for (chunkId <- 0 until 3) {
        for (peerId <- 0 until 2) {
          buffer.store(randomFloatArray(maxChunkSize), srcId = peerId, chunkId = chunkId, count = peerSize)
          buffer.reachCompletionThreshold() shouldBe false
        }
      }

      buffer.store(randomFloatArray(maxChunkSize), srcId = 2, chunkId = 0, count = peerSize)
      buffer.reachCompletionThreshold() shouldBe false

      buffer.store(randomFloatArray(maxChunkSize), srcId = 2, chunkId = 1, count = peerSize)
      buffer.reachCompletionThreshold() shouldBe true

    }

  }


  private def randomFloatArray(maxChunkSize: Int) = {
    Array.range(0, maxChunkSize).toList.map(_ => Math.abs(Random.nextFloat()) + 1).toArray
  }

}
