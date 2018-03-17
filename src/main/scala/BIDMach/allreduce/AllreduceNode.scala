package BIDMach.allreduce

import BIDMach.Learner
import BIDMach.allreduce.binder.{AllreduceBinder, ElasticAverageBinder}
import akka.actor.{Actor, ActorRef, ActorSystem, Props}
import com.typesafe.config.ConfigFactory

import scala.concurrent.duration._

/**
  * Top-level root to all-reduce actor hierarchy, with children as dimension node actors, and grandchildren as line masters/round worker actors
  * The hierarchy has the following paths;
  * for round worker [user/Node/DimensionNode-dim={}/Worker-round={}],
  * and for line master [user/Node/DimensionNode-dim={}/LineMaster]
  *
  */
class AllreduceNode(nodeConfig: NodeConfig,
                    lineMasterConfig: LineMasterConfig,
                    workerConfig: WorkerConfig,
                    binder: AllreduceBinder
                   ) extends Actor with akka.actor.ActorLogging {

  val sink = binder.dataSink
  val source = binder.dataSource
  val dimNum = nodeConfig.dimNum
  var dimensionNodeMap: Array[ActorRef] = Array.empty

  generateDimensionNodes()

  override def receive: Receive = {
    case _ => Unit
  }

  protected def dimensionNodeClassProvider(): Class[_] = {

    if (nodeConfig.reportStats) {
      classOf[AllreduceDimensionNodeWithStats]
    } else {
      classOf[AllreduceDimensionNode]
    }
  }

  def generateDimensionNodes(): Unit = {
    dimensionNodeMap = {
      val arr = new Array[ActorRef](dimNum)
      for (i <- 0 until dimNum) {
        val dimensionNode = context.actorOf(Props(
          dimensionNodeClassProvider(),
          DimensionNodeConfig(dim = i),
          lineMasterConfig,
          workerConfig,
          source,
          sink), s"DimensionNode-dim=${i}")
        println(s"-----Node: DimensionNode dim:$i created with ${dimensionNode}")
        arr(i) = dimensionNode
      }
      arr
    }
  }
}

object AllreduceNode {

  def startUp(port: String, nodeConfig: NodeConfig, lineMasterConfig: LineMasterConfig, workerConfig: WorkerConfig,
              learner: Learner) = {

    val config = ConfigFactory.parseString(s"\nakka.remote.netty.tcp.port=$port").
      withFallback(ConfigFactory.parseString("akka.cluster.roles = [Node]")).
      withFallback(ConfigFactory.load())

    val system = ActorSystem("ClusterSystem", config)

    system.actorOf(Props(classOf[AllreduceNode],
      nodeConfig,
      lineMasterConfig,
      workerConfig,
      learner
    ), name = "Node")

  }


  def getBasicConfigs() = {

    val dimNum = 2
    val maxChunkSize = 20000
    val roundWorkerPerDimNum = 3
    val maxRound = 1000000

    val threshold = ThresholdConfig(thAllreduce = 1f, thReduce = 1f, thComplete = 1f)
    val metaData = MetaDataConfig(maxChunkSize = maxChunkSize)

    val nodeConfig = NodeConfig(dimNum = dimNum, reportStats = true, elasticRate = 0.3)

    val workerConfig = WorkerConfig(
      statsReportingRoundFrequency = 5,
      threshold = threshold,
      metaData = metaData)

    val lineMasterConfig = LineMasterConfig(
      roundWorkerPerDimNum = roundWorkerPerDimNum,
      dim = -1,
      maxRound = maxRound,
      workerResolutionTimeout = 5.seconds,
      threshold = threshold,
      metaData = metaData)

    (metaData, nodeConfig, workerConfig, lineMasterConfig)

  }

  def start(learner: Learner): ActorRef = {

    val (metaData, nodeConfig, workerConfig, lineMasterConfig) = getBasicConfigs()

    val allReduceLayer = new AllreduceLayer(metaData, nodeConfig, workerConfig, lineMasterConfig)

    allReduceLayer.startAfterIter(learner, iter = 0) {
      modelMats => new ElasticAverageBinder(modelMats, nodeConfig.elasticRate)
    }

  }

  def startWithBinder(binder: AllreduceBinder): ActorRef = {

    val (metaData, nodeConfig, workerConfig, lineMasterConfig) = getBasicConfigs()

    val allReduceLayer = new AllreduceLayer(metaData, nodeConfig, workerConfig, lineMasterConfig)

    allReduceLayer.start(binder)

  }

  def getLearner(): Learner = {
    new AllreduceDummyLearner()
  }

  def main(args: Array[String]): Unit = {
    val learner = getLearner()
    learner.launchTrain
    start(learner)
  }


}

