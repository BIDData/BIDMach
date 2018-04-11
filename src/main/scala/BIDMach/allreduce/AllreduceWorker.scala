package BIDMach.allreduce


import BIDMach.allreduce.binder.AllreduceBinder.{DataSink, DataSource}
import BIDMach.allreduce.binder.{AllReduceInputRequest, AllReduceOutput}
import BIDMach.allreduce.buffer.{ReducedDataBuffer, ScatteredDataBuffer}
import akka.actor.{Actor, ActorRef}


class AllreduceWorker(config: WorkerConfig,
                      dataSource: DataSource,
                      dataSink: DataSink) extends Actor with akka.actor.ActorLogging {

  val thReduce = config.threshold.thReduce
  val thComplete = config.threshold.thComplete

  val dataSize = config.metaData.dataSize
  val maxChunkSize = config.metaData.maxChunkSize

  var currentConfig: RoundConfig = RoundConfig(-1, -1, self, Map[Int, ActorRef](), -1)
  var isCurrentRoundCompleted = true

  // Data
  var data: Array[Float] = new Array(dataSize)

  // Buffer
  var scatterBlockBuf: ScatteredDataBuffer = ScatteredDataBuffer.empty // store scattered data received
  var reduceBlockBuf: ReducedDataBuffer = ReducedDataBuffer.empty // store reduced data received

  // Output
  var output: Array[Float] = new Array(dataSize)
  var outputCount: Array[Int] = new Array(dataSize)

  println(s"\n----Worker ${self.path}")
  println(s"\n----Worker ${self.path}: Thresholds: thReduce = ${thReduce}, thComplete = ${thComplete}");

  def receive = {

    case s: StartAllreduce => {
      try {
        log.debug(s"\n----Worker ${self.path}: receive StartAllreduce, LM version = ${s.config.lineMasterVersion}, round = ${s.config.round}, workerId = ${s.config.workerId}")
        if (handleRoundConfig(s.config)) {
          isCurrentRoundCompleted = false
          fetch()
          scatter()
        }
      } catch {
        case e: Throwable => printStackTrace("start all reduce", e);
      }
    }

    case s: ScatterBlock => {
      try {
        log.debug(s"\n----Worker ${self.path}: receive scattered data from round ${s.config.round} srcId = ${s.srcId}, destId = ${s.destId}, chunkId=${s.chunkId}, myId = ${currentConfig.workerId}")
        if (handleRoundConfig(s.config)) {
          handleScatterBlock(s)
        }
      } catch {
        case e: Throwable => printStackTrace("scatter block", e);
      }
    }

    case r: ReduceBlock => {
      try {
        log.debug(s"\n----Worker ${self.path}: Receive reduced data from round ${r.config.round}, srcId = ${r.srcId}, destId = ${r.destId}, chunkId=${r.chunkId}")
        if (handleRoundConfig(r.config)) {
          handleReduceBlock(r)
        }
      } catch {
        case e: Throwable => printStackTrace("reduce block", e);
      }
    }
  }

  private def handleRoundConfig(config: RoundConfig): Boolean = {
    if (config < currentConfig) { // outdated msg, discard
      return false
    } else if (config > currentConfig) {
      if (!isCurrentRoundCompleted) { // falling behind, catch up
        catchUp()
      }
      prepareBuffer(config)
    }
    currentConfig = config
    return true
  }

  private def catchUp() {
    log.warning(s"\n----Worker ${self.path}: Force completing round ${currentConfig.round}")
    val unreducedChunkIds = scatterBlockBuf.getUnreducedChunkIds()
    for (i <- unreducedChunkIds) {
      reduceAndBroadcast(i)
    }
    completeRound()
  }

  private def requiresBufferInitialization(newConfig: RoundConfig): Boolean = {
    val newNumPeers = newConfig.peerWorkers.size
    val newScatterBlockSize = getBlockSize(newConfig.workerId, newNumPeers, dataSize)
    currentConfig.peerWorkers.size != newNumPeers || newScatterBlockSize != scatterBlockBuf.dataSize
  }

  private def prepareBuffer(config: RoundConfig) {

    if (requiresBufferInitialization(config)) { // re-initialize buffers when grid size (and thus block size) changes
      log.debug(s"\n----Worker ${self.path}: handleBuffer: reinitialize buffer")

      val newNumPeers = config.peerWorkers.size
      val newWorkerId = config.workerId

      // prepare meta-data
      val myBlockSize = getBlockSize(newWorkerId, newNumPeers, dataSize)
      val maxBlockSize = getBlockSize(0, newNumPeers, dataSize)
      val minBlockSize = getBlockSize(newNumPeers - 1, newNumPeers, dataSize)

      // reusing old implementation of buffer defaulting max lag to 1, since this is per-round worker
      log.debug(s"dataSize = ${myBlockSize}, maxBlockSize = ${myBlockSize}, minBlocksize = ${minBlockSize} peerSize = ${newNumPeers}, maxChunkSize = ${maxChunkSize}")
      scatterBlockBuf = ScatteredDataBuffer(
        dataSize = myBlockSize,
        peerSize = newNumPeers,
        reducingThreshold = thReduce,
        maxChunkSize = maxChunkSize,
        reducer = SumReducer
      )

      reduceBlockBuf = ReducedDataBuffer(
        maxBlockSize = maxBlockSize,
        minBlockSize = minBlockSize,
        totalDataSize = dataSize,
        peerSize = newNumPeers,
        completionThreshold = thComplete,
        maxChunkSize = maxChunkSize
      )
    } else { // clear buffer for new round
      scatterBlockBuf.prepareNewRound()
      reduceBlockBuf.prepareNewRound()
    }
  }

  private def handleReduceBlock(r: ReduceBlock) = {
    if (r.value.size > maxChunkSize) {
      throw new RuntimeException(s"Reduced block of size ${r.value.size} is larger than expected.. Max msg size is $maxChunkSize")
    } else if (r.destId != currentConfig.workerId) {
      throw new RuntimeException(s"Message with destination ${r.destId} was incorrectly routed to node ${currentConfig.workerId}")
    } else if (r.config.round > currentConfig.round) {
      throw new RuntimeException(s"New round ${r.config.round} should have been prepared, but current round is ${currentConfig.round}")
    }

    if (r.config.round < currentConfig.round) {
      log.debug(s"\n----Worker ${self.path}: Outdated reduced data")
    } else {
      reduceBlockBuf.store(r.value, r.srcId, r.chunkId, r.count)
      if (reduceBlockBuf.reachCompletionThreshold()) {
        log.debug(s"\n----Worker ${self.path}: Receive enough reduced data (numPeers = ${currentConfig.peerWorkers.size}) for round ${r.config.round}, complete")
        completeRound()
      }
    }
  }

  private def handleScatterBlock(s: ScatterBlock) = {
    if (s.srcId == s.destId) {
      log.debug(s"\n----Worker ${self.path}: receive scattered data from round ${s.config.round} srcId = ${s.srcId}, destId = ${s.destId}, chunkId=${s.chunkId}, myId = ${currentConfig.workerId}")
    }
    if (s.destId != currentConfig.workerId) {
      throw new RuntimeException(s"Scatter block should be directed to ${currentConfig.workerId}, but received ${s.destId}")
    } else if (s.config.round > currentConfig.round) {
      throw new RuntimeException(s"New round ${s.config.round} should have been prepared, but current round is ${currentConfig.round}")
    }

    if (s.config.round < currentConfig.round) {
      log.debug(s"\n----Worker ${self.path}: Outdated scattered data")
    } else {
      scatterBlockBuf.store(s.value, s.srcId, s.chunkId)
      if (scatterBlockBuf.reachReducingThreshold(s.chunkId)) {
        log.debug(s"\n----Worker ${self.path}: receive ${scatterBlockBuf.count(s.chunkId)} scattered data (numPeers = ${currentConfig.peerWorkers.size}), chunkId =${s.chunkId} for round ${s.config.round}, start reducing")
        reduceAndBroadcast(s.chunkId)
      }
    }

  }

  private def getBlockSize(workerId: Int, numPeers: Int, dataSize: Int): Int = {
    val blockSize = math.ceil(dataSize * 1f / numPeers).toInt
    return math.min(blockSize, dataSize - workerId * blockSize)
  }

  private def fetch() = {
    log.debug(s"\nfetch ${currentConfig.round}")
    val input = dataSource(AllReduceInputRequest(currentConfig.round))
    if (dataSize != input.data.size) {
      throw new IllegalArgumentException(s"\nInput data size ${input.data.size} is different from initialization time $dataSize!")
    }
    data = input.data
  }

  private def flush() = {
    reduceBlockBuf.getWithCounts(output, outputCount)

    dataSink(AllReduceOutput(output, outputCount, currentConfig.round))
  }

  private def scatter() = {
    val numPeers = currentConfig.peerWorkers.size
    log.debug(s"scatter: numPeers = ${numPeers}")
    for (peerId <- 0 until numPeers) {
      val idx = (peerId + currentConfig.workerId) % numPeers
      val worker = currentConfig.peerWorkers(idx)
      //Partition the dataBlock if it is too big
      val peerBlockStart = idx * getBlockSize(0, numPeers, dataSize)
      val peerBlockSize = getBlockSize(idx, numPeers, dataSize)
      val peerNumChunks = math.ceil(1f * peerBlockSize / maxChunkSize).toInt

      log.debug(s"scatter: peerNumChunks = ${peerNumChunks}")
      for (i <- 0 until peerNumChunks) {
        val chunkStart = math.min(i * maxChunkSize, peerBlockSize - 1);
        val chunkEnd = math.min((i + 1) * maxChunkSize - 1, peerBlockSize - 1);
        val chunkSize = chunkEnd - chunkStart + 1
        val chunk: Array[Float] = new Array(chunkSize)

        System.arraycopy(data, peerBlockStart + chunkStart, chunk, 0, chunkSize);
        log.debug(s"\n----Worker ${self.path}: send msg from ${currentConfig.workerId} to ${idx}, chunkId: ${i}")
        val scatterConfig = currentConfig.copy(workerId = idx)
        val scatterMsg = ScatterBlock(chunk, currentConfig.workerId, idx, i, scatterConfig)
        if (worker == self) {
          handleScatterBlock(scatterMsg)
        } else {
          sendTo(worker, scatterMsg)
        }
      }
    }
  }

  private def reduceAndBroadcast(chunkId: Int) = {
    val (reducedData, reduceCount) = scatterBlockBuf.reduce(chunkId)
    broadcast(reducedData, chunkId, reduceCount)
  }

  private def broadcast(data: Array[Float], chunkId: Int, reduceCount: Int) = {
    log.debug(s"\n----Worker ${self.path}: Start broadcasting, chunkId = ${chunkId}")
    val numPeers = currentConfig.peerWorkers.size
    for (i <- 0 until numPeers) {
      val peerWorkerId = (i + currentConfig.workerId) % numPeers
      val worker = currentConfig.peerWorkers(peerWorkerId)
      val reduceConfig = currentConfig.copy(workerId = peerWorkerId)
      val reduceMsg = ReduceBlock(data, currentConfig.workerId, peerWorkerId, chunkId, reduceConfig, reduceCount)
      if (worker == self) {
        handleReduceBlock(reduceMsg)
      } else {
        sendTo(worker, reduceMsg)
      }
    }
  }


  private def completeRound() = {
    flush()
    sendTo(currentConfig.lineMaster, CompleteAllreduce(currentConfig.workerId, currentConfig))
    isCurrentRoundCompleted = true
  }

  private def printStackTrace(location: String, e: Throwable): Unit = {
    import java.io.{PrintWriter, StringWriter}
    val sw = new StringWriter
    e.printStackTrace(new PrintWriter(sw))
    val stackTrace = sw.toString
    println(e, s"error in $location, $stackTrace")
  }

  protected def sendTo(recipient: ActorRef, msg: Any) = {
    recipient ! msg
  }

}

