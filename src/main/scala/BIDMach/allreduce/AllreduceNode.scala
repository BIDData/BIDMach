package BIDMach.allreduce

import BIDMach.allreduce.AllreduceNode.{DataSink, DataSource}
import akka.actor.{Actor, ActorRef, ActorSystem, Props}
import com.typesafe.config.ConfigFactory

import scala.concurrent.duration._


class AllreduceNode(nodeConfig: NodeConfig,
                    lineMasterConfig: LineMasterConfig,
                    workerConfig: WorkerConfig,
                    sources: List[List[DataSource]],
                    sinks: List[List[DataSink]]) extends Actor with akka.actor.ActorLogging {

  var dimensioNodeMap: Array[ActorRef] = Array.empty
  var id = -1
  var dimNum = nodeConfig.dimNum //numDim = # of DimensionNodes PlaceHolder it will spawn

  generateDimensionNodes() //generate dimension nodes when the node initializes

  override def receive: Receive = {
    case _ => Unit
  }

  def generateDimensionNodes(): Unit = {
    dimensioNodeMap = {
      val arr = new Array[ActorRef](dimNum)
      for (i <- 0 until dimNum) {
        val dimensionNode = context.actorOf(Props(
          classOf[AllreduceDimensionNode],
          DimensionNodeConfig(dim = i),
          lineMasterConfig,
          workerConfig,
          sources(i),
          sinks(i)),
          name = s"DimensionNode-dim=${i}"
        )
        println(s"-----Node: DimensionNode dim:$i created with ${dimensionNode}")
        arr(i) = dimensionNode
      }
      arr
    }
  }
}

object AllreduceNode {

  type DataSink = AllReduceOutput => Unit
  type DataSource = AllReduceInputRequest => AllReduceInput

  def startUp(port: String, nodeConfig: NodeConfig, lineMasterConfig: LineMasterConfig, workerConfig: WorkerConfig,
              assertCorrectness: Boolean = false, checkpoint: Int = 10) = {

    val config = ConfigFactory.parseString(s"\nakka.remote.netty.tcp.port=$port").
      withFallback(ConfigFactory.parseString("akka.cluster.roles = [Node]")).
      withFallback(ConfigFactory.load())

    val system = ActorSystem("ClusterSystem", config)


    def getSourceSink(): (DataSource, DataSink) = if (assertCorrectness) {
      testCorrectnessSourceSink(workerConfig.metaData.dataSize, checkpoint)
    } else {
      testPerformanceSourceSink(workerConfig.metaData.dataSize, checkpoint)
    }

    val (sourceList, sinkList) = {
      val dimSources: Array[List[DataSource]] = new Array(nodeConfig.dimNum)
      val dimSinks: Array[List[DataSink]] = new Array(nodeConfig.dimNum)
      for (i <- 0 until nodeConfig.dimNum) {
        val (source, sink) = getSourceSink()
        val sources: Array[DataSource] = Array.fill(lineMasterConfig.workerPerNodeNum)(source)
        val sinks: Array[DataSink] = Array.fill(lineMasterConfig.workerPerNodeNum)(sink)
        dimSources(i) = sources.toList
        dimSinks(i) = sinks.toList
      }
      (dimSources.toList, dimSinks.toList)
    }

    system.actorOf(Props(classOf[AllreduceNode],
      nodeConfig,
      lineMasterConfig,
      workerConfig,
      sourceList,
      sinkList
    ), name = "Node")
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

  /**
    * Test correctness of reduced data at the sink
    * @param sourceDataSize total data size
    * @param checkpoint round frequency at which the data should be checked
    */
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
    val source: DataSource = r => {
      AllReduceInput(randomFloats(r.iteration % totalInputSample))
    }

    // Specify data sink
    val sink: DataSink = r => {
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

  def main(args: Array[String]): Unit = {
    val dimNum = 2
    val dataSize = 100
    val maxChunkSize = 4
    val workerPerNodeNum = 3
    val maxRound = 100

    val threshold = ThresholdConfig(thAllreduce = 1f, thReduce = 1f, thComplete = 0.8f)
    val metaData = MetaDataConfig(dataSize = dataSize, maxChunkSize = maxChunkSize)

    val nodeConfig = NodeConfig(dimNum = dimNum)

    val workerConfig = WorkerConfig(
      discoveryTimeout = 5.seconds,
      threshold = threshold,
      metaData = metaData)

    val lineMasterConfig = LineMasterConfig(
      workerPerNodeNum = workerPerNodeNum,
      dim = -1,
      maxRound = maxRound,
      discoveryTimeout = 5.seconds,
      threshold = threshold,
      metaData = metaData)


    AllreduceNode.startUp("0", nodeConfig, lineMasterConfig, workerConfig)
  }
}

