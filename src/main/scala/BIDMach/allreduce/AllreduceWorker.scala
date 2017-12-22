package BIDMach.allreduce

import BIDMach.allreduce.buffer.{ReducedDataBuffer, ScatteredDataBuffer}
import akka.actor.{Actor, ActorRef, ActorSystem, Props, Terminated}
import com.typesafe.config.ConfigFactory

import scala.language.postfixOps

class AllreduceWorker(dataSource: AllReduceInputRequest => AllReduceInput,
                      dataSink: AllReduceOutput => Unit) extends Actor with akka.actor.ActorLogging{

  var id = -1 // node id
  var master: Option[ActorRef] = None
  var peers = Map[Int, ActorRef]() // workers in the same row/col, including self
  var peerNum = 0
  var thReduce = 1f // pct of scattered data needed to start reducing
  var thComplete = 1f // pct of reduced data needed to complete current round
  var maxLag = 0 // number of rounds allowed for a worker to fall behind
  var round = -1 // current (unfinished) round of allreduce, can potentially be maxRound+1
  var maxRound = -1 // most updated timestamp received for StartAllreduce
  var maxScattered = -1 // most updated timestamp where scatter() has been called
  var completed = Set[Int]() // set of completed rounds

  // Data
  var dataSize = 0
  var data: Array[Float] = Array.empty // store input data
  var dataRange: Array[Int] = Array.empty
  var maxBlockSize = 0
  var minBlockSize = 0
  var myBlockSize = 0
  var scatterBlockBuf: ScatteredDataBuffer = ScatteredDataBuffer.empty // store scattered data received
  var reduceBlockBuf: ReducedDataBuffer = ReducedDataBuffer.empty // store reduced data received
  var maxChunkSize = 1024; // maximum msg size that is allowed on the wire

  def receive = {

    case init: InitWorkers =>

      tryCatch("init worker") { _ =>

        if (id == -1) {
          id = init.destId
          master = Some(init.master)
          peerNum = init.workerNum
          peers = init.workers
          thReduce = init.thReduce
          thComplete = init.thComplete
          maxLag = init.maxLag
          round = 0 // clear round info to start over
          maxRound = -1
          maxScattered = -1
          completed = Set[Int]()

          dataSize = init.dataSize
          data = initArray(dataSize)
          dataRange = initDataBlockRanges()
          myBlockSize = blockSize(id)
          maxBlockSize = blockSize(0)
          minBlockSize = blockSize(peerNum - 1)

          maxChunkSize = init.maxChunkSize

          scatterBlockBuf = ScatteredDataBuffer(
            dataSize = myBlockSize,
            peerSize = peerNum,
            maxLag = maxLag + 1,
            reducingThreshold = thReduce,
            maxChunkSize = maxChunkSize
          )

          reduceBlockBuf = ReducedDataBuffer(
            maxBlockSize = maxBlockSize,
            minBlockSize = minBlockSize,
            totalDataSize = dataSize,
            peerSize = peerNum,
            maxLag = maxLag + 1,
            completionThreshold = thComplete,
            maxChunkSize = maxChunkSize
          )

          log.info(s"\n----Actor id = ${id}")
          for (i <- 0 until peers.size) {
            log.debug(s"\n----Peers[${i}] = ${peers(i)}")
          }
          println(s"----Number of peers / total peers = ${peers.size} / $peerNum")
          println(s"\n----Thresholds: thReduce = ${thReduce}, thComplete = ${thComplete}, maxLag = ${maxLag}")
          println(s"\n----Size of scatter buffer: ${scatterBlockBuf.maxLag} x ${scatterBlockBuf.peerSize} x ${scatterBlockBuf.dataSize}. threshold : ${scatterBlockBuf.minChunkRequired}")
          println(s"\n----Size of reduce buffer: ${reduceBlockBuf.maxLag} x ${reduceBlockBuf.peerSize} x ${reduceBlockBuf.maxBlockSize}. threshold: ${reduceBlockBuf.minChunkRequired}")
        } else {
          peers = init.workers
        }
      }

    case s: StartAllreduce =>
      tryCatch("start all reduce") { _ =>
        log.debug(s"\n----Start allreduce round ${s.round}")
        if (id == -1) {
          log.warning(s"\n----Actor is not initialized")
          self ! s
        } else {
          maxRound = math.max(maxRound, s.round)
          while (round < maxRound - maxLag) { // fall behind too much, catch up
            for (k <- 0 until scatterBlockBuf.numChunks) {
              val (reducedData, reduceCount) = scatterBlockBuf.reduce(0, k)
              broadcast(reducedData, k, round, reduceCount)
            }
            complete(round, 0)
          }
          while (maxScattered < maxRound) {
            fetch(maxScattered + 1)
            scatter()
            maxScattered += 1
          }
        }
        completed = completed.filterNot(e => e < round)
      }

    case s: ScatterBlock =>

      tryCatch("scatter block") { _ =>
        log.debug(s"\n----receive scattered data from round ${s.round}: value = ${s.value.toList}, srcId = ${s.srcId}, destId = ${s.destId}, chunkId=${s.chunkId}, current round = $round")
        if (id == -1) {
          log.warning(s"\n----Have not initialized!")
          self ! s
        } else {
          assert(s.destId == id)
          if (s.round < round || completed.contains(s.round)) {
            log.debug(s"\n----Outdated scattered data")
          } else if (s.round <= maxRound) {
            val row = s.round - round
            scatterBlockBuf.store(s.value, row, s.srcId, s.chunkId)
            if (scatterBlockBuf.reachReducingThreshold(row, s.chunkId)) {
              log.debug(s"\n----receive ${scatterBlockBuf.count(row, s.chunkId)} scattered data (numPeers = ${peers.size}), chunkId =${s.chunkId} for round ${s.round}, start reducing")
              val (reducedData, reduceCount) = scatterBlockBuf.reduce(row, s.chunkId)
              broadcast(reducedData, s.chunkId, s.round, reduceCount)
            }
          } else {
            self ! StartAllreduce(s.round)
            self ! s
          }
        }
      }

    case r: ReduceBlock =>

      tryCatch("reduce block") { _ =>

        log.debug(s"\n----Receive reduced data from round ${r.round}: value = ${r.value.toList}, srcId = ${r.srcId}, destId = ${r.destId}, chunkId=${r.chunkId}")
        if (id == -1) {
          log.warning(s"\n----Have not initialized!")
          self ! r
        } else {

          if (r.value.size > maxChunkSize) {
            throw new RuntimeException(s"Reduced block of size ${r.value.size} is larger than expected.. Max msg size is $maxChunkSize")
          } else if (r.destId != id) {
            throw new RuntimeException(s"Message with destination ${r.destId} was incorrectly routed to node $id")
          }
          if (r.round < round || completed.contains(r.round)) {
            log.debug(s"\n----Outdated reduced data")
          } else if (r.round <= maxRound) {
            val row = r.round - round
            reduceBlockBuf.store(r.value, row, r.srcId, r.chunkId, r.count)
            if (reduceBlockBuf.reachCompletionThreshold(row)) {
              log.debug(s"\n----Receive enough reduced data (numPeers = ${peers.size}) for round ${r.round}, complete")
              complete(r.round, row)
            }
          } else {
            self ! StartAllreduce(r.round)
            self ! r
          }
        }
      }


    case Terminated(a) =>
      for ((idx, worker) <- peers) {
        if (worker == a) {
          peers -= idx
        }
      }
  }


  private def blockSize(id: Int) = {
    val (start, end) = range(id)
    end - start
  }

  private def initArray(size: Int) = {
    Array.fill[Float](size)(0)
  }

  private def fetch(round: Int) = {
    log.debug(s"\nfetch ${round}")
    val input = dataSource(AllReduceInputRequest(round))
    if (dataSize != input.data.size) {
      throw new IllegalArgumentException(s"\nInput data size ${input.data.size} is different from initialization time $dataSize!")
    }
    data = input.data
  }

  private def flush(completedRound: Int, row: Int) = {
    val (output, counts) = reduceBlockBuf.getWithCounts(row)
    log.debug(s"\n----Flushing ${output.toList} with counts ${counts.toList} at completed round $completedRound")
    dataSink(AllReduceOutput(output, counts, completedRound))
  }

  private def scatter() = {
    for( i <- 0 until peers.size){
      val idx = (i+id) % peerNum
      peers.get(idx) match {
        case Some(worker) =>
          val dataBlock = getDataBlock(idx)
          //Partition the dataBlock if it is too big
          val peerNumChunks =  math.ceil(1f * dataBlock.length / maxChunkSize).toInt
          for (i <- 0 until peerNumChunks) {
            val chunkStart = math.min(i * maxChunkSize, dataBlock.length - 1);
            val chunkEnd = math.min((i + 1) * maxChunkSize - 1, dataBlock.length - 1);
            val chunk = new Array[Float](chunkEnd - chunkStart + 1);
            System.arraycopy(dataBlock, chunkStart, chunk, 0, chunk.length);
            log.debug(s"\n----send msg ${chunk.toList} from ${id} to ${idx}, chunkId: ${i}")
            worker ! ScatterBlock(chunk, id, idx, i, maxScattered + 1);
          }
        case None => Unit
      }
    }
  }

  private def initDataBlockRanges() = {
    val stepSize = math.ceil(dataSize * 1f / peerNum).toInt
    Array.range(0, dataSize, stepSize)
  }

  private def getDataBlock(idx: Int): Array[Float] = {
    val (start, end) = range(idx)
    val block = new Array[Float](end - start)
    System.arraycopy(data, start, block, 0, end - start)
    block
  }

  private def range(idx: Int): (Int, Int) = {
    if (idx >= peerNum - 1)
      (dataRange(idx), dataSize)
    else
      (dataRange(idx), dataRange(idx + 1))
  }

  private def broadcast(data: Array[Float], chunkId: Int, bcastRound: Int, reduceCount: Int) = {
    log.debug(s"\n----Start broadcasting")
    for( i <- 0 until peers.size){
      val idx = (i+id) % peerNum
      peers.get(idx) match {
        case Some(worker) =>
          log.debug(s"\n----Broadcast data:${data.toList}, src: ${id}, dest: ${idx}, chunkId: ${chunkId}, round: ${bcastRound}")
          worker ! ReduceBlock(data, id, idx, chunkId, bcastRound, reduceCount)
        case None => Unit
      }
    }
  }

  private def complete(completedRound: Int, row: Int) = {
    log.debug(s"\n----Complete allreduce round ${completedRound}\n")

    flush(completedRound, row)

    data = Array.empty
    master.orNull ! CompleteAllreduce(id, completedRound)
    completed = completed + completedRound
    if (round == completedRound) {
      do {
        round += 1
        scatterBlockBuf.up()
        reduceBlockBuf.up()
      } while (completed.contains(round))
    }
  }

  private def tryCatch(location: String)(work: Any => Unit): Unit = {
    try {
      work()
    } catch {
      case e: Throwable =>
        import java.io.PrintWriter
        import java.io.StringWriter
        val sw = new StringWriter
        e.printStackTrace(new PrintWriter(sw))
        val stackTrace = sw.toString
        println(e, s"error in $location, $stackTrace")
    }
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

  private def initWorker(port: String, sourceDataSize: Int, checkpoint: Int = 50) = {
    val config = ConfigFactory.parseString(s"\nakka.remote.netty.tcp.port=$port").
      withFallback(ConfigFactory.parseString("akka.cluster.roles = [worker]")).
      withFallback(ConfigFactory.load())

    val system = ActorSystem("ClusterSystem", config)

    lazy val floats = Array.range(0, sourceDataSize).map(_.toFloat)
    val source: DataSource = _ => AllReduceInput(floats)

    var tic = System.currentTimeMillis()
    val sink: DataSink = r => {
      if (r.iteration % checkpoint == 0 && r.iteration != 0) {
        val timeElapsed = (System.currentTimeMillis() - tic) / 1.0e3
        println(s"----Data output at #${r.iteration} - $timeElapsed s")
        val bytes = r.data.length * 4.0 * checkpoint
        println("%2.1f Mbytes in %2.1f seconds at %4.3f MBytes/sec" format(bytes / 1.0e6, timeElapsed, bytes / 1.0e6 / timeElapsed));
        tic = System.currentTimeMillis()
      }
    }

    system.actorOf(Props(classOf[AllreduceWorker], source, sink), name = "worker")
  }

  def startUp(port: String) = {
    main(Array(port))
  }

  def startUp(port: String, dataSize: Int, checkpoint: Int = 50) = {
    initWorker(port, dataSize, checkpoint)
  }

}