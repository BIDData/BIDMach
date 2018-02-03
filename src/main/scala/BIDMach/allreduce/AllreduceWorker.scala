package BIDMach.allreduce


import BIDMach.allreduce.buffer.{ReducedDataBuffer, ScatteredDataBuffer}
import akka.actor.{Actor, ActorRef, Terminated}


class AllreduceWorker(config: WorkerConfig,
                      dataSource: AllReduceInputRequest => AllReduceInput,
                      dataSink: AllReduceOutput => Unit) extends Actor with akka.actor.ActorLogging {

  val thReduce = config.threshold.thReduce
  val thComplete = config.threshold.thComplete

  val dataSize = config.metaData.dataSize
  val maxChunkSize = config.metaData.maxChunkSize

  val workerDiscoveryTimeout = config.discoveryTimeout

  var master: Option[ActorRef] = None
  var workerPeers = Map[Int, ActorRef]() // workers of the same round across other the nodes
  var workerPeerNum = 0

  var workerId = -1
  var currentRound = -1
  var isCompleted = true

  // Data
  var data: Array[Float] = new Array(dataSize)
  var dataRange: Array[Int] = Array.empty
  var maxBlockSize = 0
  var minBlockSize = 0
  var myBlockSize = 0

  // Buffer
  var scatterBlockBuf: ScatteredDataBuffer = ScatteredDataBuffer.empty // store scattered data received
  var reduceBlockBuf: ReducedDataBuffer = ReducedDataBuffer.empty // store reduced data received

  // Output
  var output: Array[Float] = new Array(dataSize)
  var outputCount: Array[Int] = new Array(dataSize)

  println(s"\n----Worker ${self.path}")
  println(s"\n----Worker ${self.path}: Thresholds: thReduce = ${thReduce}, thComplete = ${thComplete}");

  def receive = {

    case p: PrepareAllreduce => {

      log.debug(s"\n----Worker ${self.path}: Preparing data round ${p.round}")
      try {

        assert(p.round > currentRound)

        if (!isCompleted) {
          log.warning(s"\n----Worker ${self.path}: Force completing round ${p.round}")
          val unreducedChunkIds = scatterBlockBuf.getUnreducedChunkIds()
          for (i <- unreducedChunkIds) {
            reduceAndBroadcast(i)
          }
          completeRound()
        }

        // TODO: to reconsider potential bugs of changing master
        // peer organization
        master = Some(sender())
        workerId = p.nodeId

        // avoid re-initialization when grid organization doesn't change (i.e. block size doesn't change)
        if (p.workerAddresses.size != workerPeerNum || p.nodeId != workerId) {
          workerPeers = p.workerAddresses
          workerPeerNum = p.workerAddresses.size

          // prepare meta-data
          dataRange = initDataBlockRanges()
          myBlockSize = blockSize(workerId)
          maxBlockSize = blockSize(0)
          minBlockSize = blockSize(workerPeerNum - 1)

          // reusing old implementation of buffer defaulting max lag to 1, since this is per-round worker
          scatterBlockBuf = ScatteredDataBuffer(
            dataSize = myBlockSize,
            peerSize = workerPeerNum,
            reducingThreshold = thReduce,
            maxChunkSize = maxChunkSize
          )

          reduceBlockBuf = ReducedDataBuffer(
            maxBlockSize = maxBlockSize,
            minBlockSize = minBlockSize,
            totalDataSize = dataSize,
            peerSize = workerPeerNum,
            completionThreshold = thComplete,
            maxChunkSize = maxChunkSize
          )
        } else {
          scatterBlockBuf.prepareNewRound()
          reduceBlockBuf.prepareNewRound()
        }

        // prepare state for new round
        currentRound = p.round
        isCompleted = false

        // acknowledge preparation done
        master.orNull ! ConfirmPreparation(p.round)

      } catch {
        case e: Throwable => printStackTrace("prepare block", e);
      }
    }

    case s: StartAllreduce => {
      try {
        assert(s.round == currentRound)

        fetch()
        scatter()

      } catch {
        case e: Throwable => printStackTrace("start all reduce", e);
      }
    }

    case s: ScatterBlock => {
      try {
        log.debug(s"\n----Worker ${self.path}: receive scattered data from round ${s.round} srcId = ${s.srcId}, destId = ${s.destId}, chunkId=${s.chunkId}")
        handleScatterBlock(s);
      } catch {
        case e: Throwable => printStackTrace("scatter block", e);
      }
    }

    case r: ReduceBlock => {
      try {
        log.debug(s"\n----Worker ${self.path}: Receive reduced data from round ${r.round}, srcId = ${r.srcId}, destId = ${r.destId}, chunkId=${r.chunkId}")
        handleReduceBlock(r);
      } catch {
        case e: Throwable => printStackTrace("reduce block", e);
      }
    }

    case Terminated(a) =>
      for ((idx, worker) <- workerPeers) {
        if (worker == a) {
          workerPeers -= idx
        }
      }
  }

  private def handleReduceBlock(r: ReduceBlock) = {
    if (r.value.size > maxChunkSize) {
      throw new RuntimeException(s"Reduced block of size ${r.value.size} is larger than expected.. Max msg size is $maxChunkSize")
    } else if (r.destId != workerId) {
      throw new RuntimeException(s"Message with destination ${r.destId} was incorrectly routed to node $workerId")
    } else if (r.round > currentRound) {
      throw new RuntimeException(s"New round ${r.round} should have been prepared, but current round is $currentRound")
    }

    if (r.round < currentRound) {
      log.debug(s"\n----Worker ${self.path}: Outdated reduced data")
    } else {
      reduceBlockBuf.store(r.value, r.srcId, r.chunkId, r.count)
      if (reduceBlockBuf.reachCompletionThreshold()) {
        log.debug(s"\n----Worker ${self.path}: Receive enough reduced data (numPeers = ${workerPeers.size}) for round ${r.round}, complete")
        completeRound()
      }
    }
  }

  private def handleScatterBlock(s: ScatterBlock) = {

    if (s.destId != workerId) {
      throw new RuntimeException(s"Scatter block should be directed to $workerId, but received ${s.destId}")
    } else if (s.round > currentRound) {
      throw new RuntimeException(s"New round ${s.round} should have been prepared, but current round is $currentRound")
    }

    if (s.round < currentRound) {
      log.debug(s"\n----Worker ${self.path}: Outdated scattered data")
    } else {
      scatterBlockBuf.store(s.value, s.srcId, s.chunkId)
      if (scatterBlockBuf.reachReducingThreshold(s.chunkId)) {
        log.debug(s"\n----Worker ${self.path}: receive ${scatterBlockBuf.count(s.chunkId)} scattered data (numPeers = ${workerPeers.size}), chunkId =${s.chunkId} for round ${s.round}, start reducing")
        reduceAndBroadcast(s.chunkId)
      }
    }

  }

  private def blockSize(id: Int) = {
    val (start, end) = range(id)
    end - start
  }

  private def initDataBlockRanges() = {
    val stepSize = math.ceil(dataSize * 1f / workerPeerNum).toInt
    Array.range(0, dataSize, stepSize)
  }

  private def range(idx: Int): (Int, Int) = {
    if (idx >= workerPeerNum - 1)
      (dataRange(idx), dataSize)
    else
      (dataRange(idx), dataRange(idx + 1))
  }

  private def fetch() = {
    log.debug(s"\nfetch ${currentRound}")
    val input = dataSource(AllReduceInputRequest(currentRound))
    if (dataSize != input.data.size) {
      throw new IllegalArgumentException(s"\nInput data size ${input.data.size} is different from initialization time $dataSize!")
    }
    data = input.data
  }

  private def flush() = {
    reduceBlockBuf.getWithCounts(output, outputCount)
    log.debug(s"\n----Worker ${self.path}: Flushing output at completed round $currentRound")
    dataSink(AllReduceOutput(output, outputCount, currentRound))
  }

  private def scatter() = {
    for (peerId <- 0 until workerPeerNum) {
      val idx = (peerId + workerId) % workerPeerNum
      val worker = workerPeers(idx)
      //Partition the dataBlock if it is too big
      val (blockStart, blockEnd) = range(idx)
      val peerBlockSize = blockEnd - blockStart
      val peerNumChunks = math.ceil(1f * peerBlockSize / maxChunkSize).toInt
      for (i <- 0 until peerNumChunks) {
        val chunkStart = math.min(i * maxChunkSize, peerBlockSize - 1);
        val chunkEnd = math.min((i + 1) * maxChunkSize - 1, peerBlockSize - 1);
        val chunkSize = chunkEnd - chunkStart + 1
        val chunk: Array[Float] = new Array(chunkSize)

        System.arraycopy(data, blockStart + chunkStart, chunk, 0, chunkSize);
        log.debug(s"\n----Worker ${self.path}: send msg from ${workerId} to ${idx}, chunkId: ${i}")
        val scatterMsg = ScatterBlock(chunk, workerId, idx, i, currentRound)
        if (worker == self) {
          handleScatterBlock(scatterMsg)
        } else {
          worker ! scatterMsg
        }
      }
    }
  }

  private def reduceAndBroadcast(chunkId: Int) = {
    val (reducedData, reduceCount) = scatterBlockBuf.reduce(chunkId)
    broadcast(reducedData, chunkId, reduceCount)
  }

  private def broadcast(data: Array[Float], chunkId: Int, reduceCount: Int) = {
    log.debug(s"\n----Worker ${self.path}: Start broadcasting")
    for (i <- 0 until workerPeerNum) {
      val peerworkerId = (i + workerId) % workerPeerNum
      val worker = workerPeers(peerworkerId)
      log.debug(s"\n----Worker ${self.path}: Broadcast reduced block src: ${workerId}, dest: ${peerworkerId}, chunkId: ${chunkId}, round: ${currentRound}")
      val reduceMsg = ReduceBlock(data, workerId, peerworkerId, chunkId, currentRound, reduceCount)
      if (worker == self) {
        handleReduceBlock(reduceMsg)
      } else {
        worker ! reduceMsg
      }
    }
  }

  private def completeRound() = {
    flush()
    master.orNull ! CompleteAllreduce(workerId, currentRound)
    isCompleted = true
  }

  private def printStackTrace(location: String, e: Throwable): Unit = {
    import java.io.{PrintWriter, StringWriter}
    val sw = new StringWriter
    e.printStackTrace(new PrintWriter(sw))
    val stackTrace = sw.toString
    println(e, s"error in $location, $stackTrace")
  }
}

