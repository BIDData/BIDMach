package BIDMach.allreduce


import BIDMach.allreduce.buffer.{ReducedDataBuffer, ScatteredDataBuffer}
import akka.actor.{Actor, ActorRef, ActorSystem, Props, Terminated}
import com.typesafe.config.ConfigFactory

import scala.collection.mutable


class AllreduceWorker(dataSource: AllReduceInputRequest => AllReduceInput,
                      dataSink: AllReduceOutput => Unit) extends Actor with akka.actor.ActorLogging {

  var id = -1 // node id
  var master: Option[ActorRef] = None
  var peers = Map[Int, ActorRef]() // workers in the same row/col, including self
  var peerNum = 0
  var thReduce = 1f // pct of scattered data needed to start reducing
  var thComplete = 1f // pct of reduced data needed to complete current round
  var maxLag = 0 // number of rounds allowed for a worker to fall behind
  var maxRound = -1 // most updated timestamp received for StartAllreduce
  var maxScattered = -1 // most updated timestamp where scatter() has been called

  // Data
  var dataSize = 0
  var data: Array[Float] = Array.empty // store input data
  var dataRange: Array[Int] = Array.empty
  var maxBlockSize = 0
  var minBlockSize = 0
  var myBlockSize = 0
  var maxChunkSize = 0; // maximum msg size that is allowed on the wire
  var scatterBlockBuf: ScatteredDataBuffer = ScatteredDataBuffer.empty // store scattered data received
  var reduceBlockBuf: ReducedDataBuffer = ReducedDataBuffer.empty // store reduced data received

  // Output
  var output: Array[Float] = Array.empty //
  var outputCount: Array[Int] = Array.empty //

  def receive = {

    case init: InitWorkers => {
      try {
        if (id == -1) {
          id = init.destId;
          master = Some(init.master);
          peerNum = init.workerNum;
          peers = init.workers;
          thReduce = init.thReduce;
          thComplete = init.thComplete;
          maxLag = init.maxLag;
          maxRound = -1;
          maxScattered = -1;

          dataSize = init.dataSize;
          data = new Array(dataSize);
          dataRange = initDataBlockRanges();
          myBlockSize = blockSize(id);
          maxBlockSize = blockSize(0);
          minBlockSize = blockSize(peerNum - 1);

          output = new Array(dataSize)
          outputCount = new Array(dataSize)

          maxChunkSize = init.maxChunkSize;

          scatterBlockBuf = ScatteredDataBuffer(
            dataSize = myBlockSize,
            peerSize = peerNum,
            maxLag = maxLag + 1,
            reducingThreshold = thReduce,
            maxChunkSize = maxChunkSize
          );

          reduceBlockBuf = ReducedDataBuffer(
            maxBlockSize = maxBlockSize,
            minBlockSize = minBlockSize,
            totalDataSize = dataSize,
            peerSize = peerNum,
            maxLag = maxLag + 1,
            completionThreshold = thComplete,
            maxChunkSize = maxChunkSize
          );

          log.info(s"\n----Actor id = ${id}")
          for (i <- 0 until peers.size) {
            log.debug(s"\n----Peers[${i}] = ${peers.get(i)}")
          }
          println(s"----Number of peers / total peers = ${peers.size} / $peerNum");
          println(s"\n----Thresholds: thReduce = ${thReduce}, thComplete = ${thComplete}, maxLag = ${maxLag}");
          println(s"\n----Size of scatter buffer: ${scatterBlockBuf.maxLag} x ${scatterBlockBuf.peerSize} x ${scatterBlockBuf.dataSize}. threshold : ${scatterBlockBuf.minChunkRequired}")
          println(s"\n----Size of reduce buffer: ${reduceBlockBuf.maxLag} x ${reduceBlockBuf.peerSize} x ${reduceBlockBuf.maxBlockSize}. threshold: ${reduceBlockBuf.minChunkRequired}")
        } else {
          peers = init.workers
        }
      } catch {
        case e: Throwable => printStackTrace("init worker", e)
      }
    }

    case s: StartAllreduce => {
      try {
        log.debug(s"\n----Start allreduce round ${s.round}");
        if (id == -1) {
          self ! s;
        } else {
          maxRound = math.max(maxRound, s.round);
          while (maxScattered < maxRound) {
            fetch(maxScattered + 1);
            scatter(maxScattered + 1);
            maxScattered += 1;
          }
        }
      } catch {
        case e: Throwable => printStackTrace("start all reduce", e);
      }
    }

    case s: ScatterBlock => {
      try {
        log.debug(s"\n----receive scattered data from round ${s.round} srcId = ${s.srcId}, destId = ${s.destId}, chunkId=${s.chunkId}")
        if (id == -1) {
          self ! s;
        } else {
          handleScatterBlock(s);
        }
      } catch {
        case e: Throwable => printStackTrace("scatter block", e);
      }
    }

    case r: ReduceBlock => {
      try {
        log.debug(s"\n----Receive reduced data from round ${r.round}, srcId = ${r.srcId}, destId = ${r.destId}, chunkId=${r.chunkId}")
        if (id == -1) {
          self ! r;
        } else {
          handleReduceBlock(r);
        }
      } catch {
        case e: Throwable => printStackTrace("reduce block", e);
      }
    }


    case Terminated(a) =>
      for ((idx, worker) <- peers) {
        if (worker == a) {
          peers -= idx
        }
      }
  }

  private def handleReduceBlock(r: ReduceBlock) = {
    if (r.value.size > maxChunkSize) {
      throw new RuntimeException(s"Reduced block of size ${r.value.size} is larger than expected.. Max msg size is $maxChunkSize")
    } else if (r.destId != id) {
      throw new RuntimeException(s"Message with destination ${r.destId} was incorrectly routed to node $id")
    }

    if (r.round > maxRound) {
      self ! StartAllreduce(r.round)
      self ! r
    } else {
      val comparison = reduceBlockBuf.compareRoundTo(r.round)
      if (comparison > 0) {
        log.debug(s"\n----Outdated reduced data")
      } else {
        if (comparison < 0) {
          val outdatedRound = reduceBlockBuf.getRound(r.round)
          completeRound(outdatedRound)
        }
        reduceBlockBuf.store(r.value, r.round, r.srcId, r.chunkId, r.count)
        if (reduceBlockBuf.reachCompletionThreshold(r.round)) {
          log.debug(s"\n----Receive enough reduced data (numPeers = ${peers.size}) for round ${r.round}, complete")
          completeRound(r.round)
        }
      }
    }

  }

  private def handleScatterBlock(s: ScatterBlock) = {
    if (s.destId != id) {
      throw new RuntimeException(s"Scatter block should be directed to $id, but received ${s.destId}")
    }

    if (s.round > maxRound) {
      self ! StartAllreduce(s.round)
      self ! s
    } else {
      val comparison = scatterBlockBuf.compareRoundTo(s.round, s.chunkId)
      if (comparison > 0) {
        log.debug(s"\n----Outdated scattered data")
      } else {
        if (comparison < 0) {
          val outdatedRound = scatterBlockBuf.getRound(s.round, s.chunkId)
          reduceAndBroadcast(outdatedRound, s.chunkId)
        }
        scatterBlockBuf.store(s.value, s.round, s.srcId, s.chunkId)
        if (scatterBlockBuf.reachReducingThreshold(s.round, s.chunkId)) {
          log.debug(s"\n----receive ${scatterBlockBuf.count(s.round, s.chunkId)} scattered data (numPeers = ${peers.size}), chunkId =${s.chunkId} for round ${s.round}, start reducing")
          reduceAndBroadcast(s.round, s.chunkId)
        }
      }
    }
  }

  private def blockSize(id: Int) = {
    val (start, end) = range(id)
    end - start
  }

  private def initDataBlockRanges() = {
    val stepSize = math.ceil(dataSize * 1f / peerNum).toInt
    Array.range(0, dataSize, stepSize)
  }

  private def range(idx: Int): (Int, Int) = {
    if (idx >= peerNum - 1)
      (dataRange(idx), dataSize)
    else
      (dataRange(idx), dataRange(idx + 1))
  }


  private def fetch(round: Int) = {
    log.debug(s"\nfetch ${round}")
    val input = dataSource(AllReduceInputRequest(round))
    if (dataSize != input.data.size) {
      throw new IllegalArgumentException(s"\nInput data size ${input.data.size} is different from initialization time $dataSize!")
    }
    data = input.data
  }

  private def flush(completedRound: Int) = {
    reduceBlockBuf.getWithCounts(completedRound, output, outputCount)
    log.debug(s"\n----Flushing output at completed round $completedRound")
    dataSink(AllReduceOutput(output, outputCount, completedRound))
  }

  private def scatter(round: Int) = {
    for (i <- 0 until peerNum) {
      val idx = (i + id) % peerNum
      peers.get(idx) match {
        case Some(worker) =>
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
            log.debug(s"\n----send msg from ${id} to ${idx}, chunkId: ${i}")
            val scatterMsg = ScatterBlock(chunk, id, idx, i, round)
            if (worker == self) {
              handleScatterBlock(scatterMsg)
            } else {
              worker ! scatterMsg
            }
          }
        case None => Unit
      }
    }
  }

  private def reduceAndBroadcast(round: Int, chunkId: Int) = {
    val (reducedData, reduceCount) = scatterBlockBuf.reduce(round, chunkId)
    broadcast(reducedData, chunkId, round, reduceCount)
    scatterBlockBuf.prepareNewRound(round, chunkId)
  }

  private def broadcast(data: Array[Float], chunkId: Int, bcastRound: Int, reduceCount: Int) = {
    log.debug(s"\n----Start broadcasting")
    for (i <- 0 until peerNum) {
      val idx = (i + id) % peerNum
      peers.get(idx) match {
        case Some(worker) =>
          log.debug(s"\n----Broadcast reduced block src: ${id}, dest: ${idx}, chunkId: ${chunkId}, round: ${bcastRound}")
          val reduceMsg = ReduceBlock(data, id, idx, chunkId, bcastRound, reduceCount)
          if (worker == self) {
            handleReduceBlock(reduceMsg)
          } else {
            worker ! reduceMsg
          }
        case None => Unit
      }
    }
  }

  private def completeRound(completedRound: Int) = {
    log.debug(s"\n----Complete allreduce round ${completedRound}\n")
    flush(completedRound)
    master.orNull ! CompleteAllreduce(id, completedRound)
    reduceBlockBuf.prepareNewRound(completedRound)
  }

  private def printStackTrace(location: String, e: Throwable): Unit = {
    import java.io.PrintWriter
    import java.io.StringWriter
    val sw = new StringWriter
    e.printStackTrace(new PrintWriter(sw))
    val stackTrace = sw.toString
    println(e, s"error in $location, $stackTrace")
  }
}

object AllreduceWorker {

  type DataSink = AllReduceOutput => Unit
  type DataSource = AllReduceInputRequest => AllReduceInput


  def main(args: Array[String]): Unit = {
    val port = if (args.isEmpty) "2553" else args(0)
    val sourceDataSize = if (args.length <= 1) 10 else args(1).toInt

    initWorker(port, sourceDataSize)
  }

  private def initWorker(port: String, sourceDataSize: Int, checkpoint: Int = 50, assertCorrectness: Boolean = false) = {
    val config = ConfigFactory.parseString(s"\nakka.remote.netty.tcp.port=$port").
      withFallback(ConfigFactory.parseString("akka.cluster.roles = [worker]")).
      withFallback(ConfigFactory.load())

    val system = ActorSystem("ClusterSystem", config)

    val (source, sink) = if (assertCorrectness) {
      testCorrectnessSourceSink(sourceDataSize, checkpoint)
    } else {
      testPerformanceSourceSink(sourceDataSize, checkpoint)
    }

    system.actorOf(Props(classOf[AllreduceWorker], source, sink), name = "worker")
  }

  private def testCorrectnessSourceSink(sourceDataSize: Int, checkpoint: Int) = {

    val random = new scala.util.Random(100)
    val totalInputSample = 8

    lazy val randomFloats = {
      val nestedArray = new Array[Array[Float]](totalInputSample)
      for (i <- 0 until totalInputSample) {
        nestedArray(i) = Array.range(0, sourceDataSize).toList.map(_ => random.nextFloat()).toArray
      }
      nestedArray
    }

    def ~=(x: Double, y: Double, precision: Double = 1e-5) = {
      if ((x - y).abs < precision) true else false
    }

    // Specify data source
    val inputSet = mutable.HashSet[Int]()
    val source: DataSource = r => {
      assert(!inputSet.contains(r.iteration), s"Same data ${r.iteration} is being requested more than once")
      inputSet.add(r.iteration)
      AllReduceInput(randomFloats(r.iteration % totalInputSample))
    }

    // Specify data sink
    val outputSet = mutable.HashSet[Int]()

    val sink: DataSink = r => {
      assert(!outputSet.contains(r.iteration), s"Output data ${r.iteration} is being flushed more than once")
      outputSet.add(r.iteration)

      if (r.iteration % checkpoint == 0) {
        val inputUsed = randomFloats(r.iteration % totalInputSample)
        println(s"\n----Asserting #${r.iteration} output...")
        var zeroCountNum = 0
        var totalCount = 0
        for (i <- 0 until sourceDataSize) {
          val count = r.count(i)
          val meanActual = r.data(i) / count
          totalCount += count
          if (count == 0) {
            zeroCountNum += 1
          } else {
            val expected = inputUsed(i)
            assert(~=(expected, meanActual), s"Expected [$expected], but actual [$meanActual] at pos $i for iteraton #${r.iteration}")
          }
        }
        val nonZeroCountElementNum = sourceDataSize - zeroCountNum
        println("OK: Mean of non-zero elements match the expected input!")
        println(f"Element with non-zero counts: ${nonZeroCountElementNum / sourceDataSize.toFloat}%.2f ($nonZeroCountElementNum/$sourceDataSize)")
        println(f"Average count value: ${totalCount / nonZeroCountElementNum.toFloat}%2.2f ($totalCount/$nonZeroCountElementNum)")
      }
    }

    (source, sink)
  }


  private def testPerformanceSourceSink(sourceDataSize: Int, checkpoint: Int): (DataSource, DataSink) = {

    lazy val floats = Array.range(0, sourceDataSize).map(_.toFloat)
    val source: DataSource = _ => AllReduceInput(floats)

    var cumulativeThroughput: Double = 0
    var measurementCount: Int = 0
    val initialDiscard: Int = 10

    var tic = System.currentTimeMillis()
    val sink: DataSink = r => {
      if (r.iteration % checkpoint == 0 && r.iteration != 0) {

        val timeElapsed = (System.currentTimeMillis() - tic) / 1.0e3

        println(s"----Data output at #${r.iteration} - $timeElapsed s")
        val bytes = r.data.length * 4.0 * checkpoint
        val mBytes = bytes / 1.0e6
        val throughput = mBytes / timeElapsed

        val report = f"$mBytes%2.1f Mbytes in $timeElapsed%2.1f seconds at $throughput%4.3f MBytes/sec"

        measurementCount += 1

        val avgReport = if (measurementCount > initialDiscard) {
          cumulativeThroughput += throughput
          val effectiveCount = measurementCount - initialDiscard
          val avgThroughput = cumulativeThroughput / effectiveCount
          f", mean throughput at $avgThroughput%4.3f MBytes/sec from $effectiveCount samples"
        } else ""

        println(s"$report$avgReport")

        tic = System.currentTimeMillis()
      }
    }


    (source, sink)
  }


  def startUp(port: String) = {
    main(Array(port))
  }

  /**
    * Test start up method
    *
    * @param port              port number
    * @param dataSize          number of elements in input array from each node to be reduced
    * @param checkpoint        interval at which timing is calculated
    * @param assertCorrectness expected multiple of input as reduced results
    * @return
    */
  def startUp(port: String, dataSize: Int, checkpoint: Int = 50, assertCorrectness: Boolean = false) = {
    initWorker(port, dataSize, checkpoint, assertCorrectness)
  }

}