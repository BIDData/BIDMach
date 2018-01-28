package BIDMach.allreduce

import BIDMach.allreduce.AllreduceWorker.{DataSink, DataSource}
import akka.actor.{Actor, ActorRef, ActorSystem, Props}
import com.typesafe.config.ConfigFactory
import scala.collection.mutable.ArrayBuffer



class AllreduceNode(nodeConfig: NodeConfig,
                    lineMasterConfig: LineMasterConfig,
                    workerConfig: WorkerConfig,
                    sources: List[DataSource],
                    sinks: List[DataSink]) extends Actor with akka.actor.ActorLogging {

  var dimensioNodeMap : Array[ActorRef] = Array.empty
  var id = -1
  var dimNum = nodeConfig.dimNum //numDim = # of DimensionNodes PlaceHolder it will spawn

  generateDimensionNodes()   //generate dimension nodes when the node initializes

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
            sources,
            sinks),
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

  def startUp(port: String, nodeConfig: NodeConfig, lineMasterConfig: LineMasterConfig, workerConfig: WorkerConfig) = {

    val config = ConfigFactory.parseString(s"\nakka.remote.netty.tcp.port=$port").
      withFallback(ConfigFactory.parseString("akka.cluster.roles = [Node]")).
      withFallback(ConfigFactory.load())

    val system = ActorSystem("ClusterSystem", config)

    lazy val floats = Array.range(0, workerConfig.metaData.dataSize).map(_.toFloat)
    val source: DataSource = _ => AllReduceInput(floats)
    val sink: DataSink = _ => Unit

    val sources: List[DataSource] = Array.fill(lineMasterConfig.roundNum * nodeConfig.dimNum)(source).toList
    val sinks:List[DataSink] = Array.fill(lineMasterConfig.roundNum * nodeConfig.dimNum)(sink).toList

    system.actorOf(Props(classOf[AllreduceNode],
      nodeConfig,
      lineMasterConfig,
      workerConfig,
      sources,
      sinks
    ), name = "Node")
  }
}

