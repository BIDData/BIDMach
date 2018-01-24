package BIDMach.allreduce

import BIDMach.allreduce.AllreduceWorker.{DataSink, DataSource}
import akka.actor.{Actor, ActorRef, ActorSystem, Props}
import com.typesafe.config.ConfigFactory

import scala.concurrent.duration._
import scala.collection.mutable

class AllreduceNode(workerConfig: WorkerConfig,
                    sources: List[DataSource],
                    sinks: List[DataSink]) extends Actor with akka.actor.ActorLogging {


  val workers: Array[ActorRef] = {
    val arr = new Array[ActorRef](workerConfig.workerPerNodeNum)
    for (i <- 0 until workerConfig.workerPerNodeNum) {
      val worker = context.actorOf(Props(
        classOf[AllreduceWorker],
        workerConfig,
        sources(i),
        sinks(i)),
        name = s"worker-$i"
      )
      println(s"Worker $i created with ${worker.path}")
      arr(i) = worker
    }
    arr
  }

  override def receive: Receive = {
    case _ => Unit
  }
}

object AllreduceNode {


  def startUp(port: String, workerConfig: WorkerConfig) = {

    val config = ConfigFactory.parseString(s"\nakka.remote.netty.tcp.port=$port").
      withFallback(ConfigFactory.parseString("akka.cluster.roles = [node]")).
      withFallback(ConfigFactory.load())

    val system = ActorSystem("ClusterSystem", config)

    val assertCorrectness = true
    val checkpoint = 10

    def testPerformanceSourceSink(sourceDataSize: Int, checkpoint: Int): (DataSource, DataSink) = {

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

    def testCorrectnessSourceSink(sourceDataSize: Int, checkpoint: Int) = {

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

    val (source, sink) = if (assertCorrectness) {
      testCorrectnessSourceSink(workerConfig.metaData.dataSize, checkpoint)
    } else {
      testPerformanceSourceSink(workerConfig.metaData.dataSize, checkpoint)
    }

    val sources: List[DataSource] = Array.fill(workerConfig.workerPerNodeNum)(source).toList
    val sinks:List[DataSink] = Array.fill(workerConfig.workerPerNodeNum)(sink).toList

    system.actorOf(Props(classOf[AllreduceNode],
      workerConfig,
      sources,
      sinks
    ), name = "node")

  }

  def main(args: Array[String]): Unit = {
    val workerPerNodeNum = 3
    val dataSize = 500000

    val maxChunkSize = 20000

    val threshold = ThresholdConfig(thAllreduce = 0.5f, thReduce = 0.5f, thComplete = 0.5f)
    val metaData = MetaDataConfig(dataSize = dataSize, maxChunkSize = maxChunkSize)

    val workerConfig = WorkerConfig(workerPerNodeNum = workerPerNodeNum,
      discoveryTimeout = 5.seconds,
      threshold = threshold,
      metaData= metaData)

    AllreduceNode.startUp("0", workerConfig)
  }


}