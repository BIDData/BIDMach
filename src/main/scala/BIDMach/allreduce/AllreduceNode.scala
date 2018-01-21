package BIDMach.allreduce

import BIDMach.allreduce.AllreduceWorker.{DataSink, DataSource}
import akka.actor.{Actor, ActorRef, ActorSystem, Props}
import com.typesafe.config.ConfigFactory

import scala.concurrent.duration._

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

    lazy val floats = Array.range(0, workerConfig.metaData.dataSize).map(_.toFloat)
    val source: DataSource = _ => AllReduceInput(floats)
    val sink: DataSink = _ => Unit

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
    val dataSize = 100

    val maxChunkSize = 4

    val threshold = ThresholdConfig(thAllreduce = 1f, thReduce = 1f, thComplete = 0.8f)
    val metaData = MetaDataConfig(dataSize = dataSize, maxChunkSize = maxChunkSize)

    val workerConfig = WorkerConfig(workerPerNodeNum = workerPerNodeNum,
      discoveryTimeout = 5.seconds,
      threshold = threshold,
      metaData= metaData)

    AllreduceNode.startUp("0", workerConfig)
  }


}