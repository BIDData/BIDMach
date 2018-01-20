package BIDMach.allreduce

import BIDMach.allreduce.AllreduceWorker.{DataSink, DataSource}
import akka.actor.{Actor, ActorRef, ActorSystem, Props}
import com.typesafe.config.ConfigFactory


class AllreduceNode(workerConfig: WorkerConfig,
                    sources: List[DataSource],
                    sinks: List[DataSink]) extends Actor with akka.actor.ActorLogging {


  val workers: Array[ActorRef] = {
    val arr = new Array[ActorRef](workerConfig.workerNum)
    for (i <- 0 until workerConfig.workerNum) {
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

    val sources: List[DataSource] = Array.fill(workerConfig.workerNum)(source).toList
    val sinks:List[DataSink] = Array.fill(workerConfig.workerNum)(sink).toList

    system.actorOf(Props(classOf[AllreduceNode],
      workerConfig,
      sources,
      sinks
    ), name = "node")

  }


}