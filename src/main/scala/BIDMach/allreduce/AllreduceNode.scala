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
                    binder: AllreduceBinder
                   ) extends Actor with akka.actor.ActorLogging {

  val sink = binder.dataSink
  val source = binder.dataSource
  val dimNum = nodeConfig.dimNum
  var dimensionNodeMap: Array[ActorRef] = Array.empty

  generateDimensionNodes()

  override def receive: Receive = {
    case StopAllreduceNode => context.stop(self)
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
          nodeConfig.lineMasterConfig,
          nodeConfig.workerConfig,
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

  def getBasicConfigs() : NodeConfig = {

    val dimNum = 2
    val maxChunkSize = 20000
    val roundWorkerPerDimNum = 3
    val maxRound = 1000000

    val threshold = ThresholdConfig(thAllreduce = 1f, thReduce = 1f, thComplete = 1f)
    val metaData = MetaDataConfig(maxChunkSize = maxChunkSize)

    val workerConfig = WorkerConfig(
      statsReportingRoundFrequency = 5,
      threshold = threshold,
      metaData = metaData)

    val lineMasterConfig = LineMasterConfig(
      roundWorkerPerDimNum = roundWorkerPerDimNum,
      dim = -1,
      maxRound = maxRound,
      workerResolutionTimeout = 5.seconds,
      threshold = threshold)

    NodeConfig(workerConfig, lineMasterConfig, dimNum = dimNum, reportStats = true, elasticRate = 0.3f)
  }

  def startAllreduceNode(binder: AllreduceBinder, nodeConfig: NodeConfig): ActorRef = {

    val config = ConfigFactory.parseString(s"akka.remote.netty.tcp.port=0").
      withFallback(ConfigFactory.parseString("akka.cluster.roles = [Node]")).
      withFallback(ConfigFactory.load())

    val updatedNodeConig = nodeConfig.copy(
      workerConfig = nodeConfig.workerConfig.copy(
        metaData = nodeConfig.workerConfig.metaData.copy(
          dataSize = binder.totalDataSize
        )
      )
    )

    val system = ActorSystem("ClusterSystem", config)
    val nodeRef = system.actorOf(Props(classOf[AllreduceNode],
      updatedNodeConig,
      binder
    ), name = "Node")
    nodeRef
  }

  def startNodeAfterIter(learner: Learner, iter: Int, nodeConfig: NodeConfig, binder: AllreduceBinder): ActorRef = {
    def createAllReduceNode(): Option[ActorRef] = {
      if (learner.synchronized(learner.ipass > 0 || learner.istep > iter)) {
        val allReduceNode = startAllreduceNode(binder, nodeConfig)
        Some(allReduceNode)
      } else {
        println(s"Learner is still at #pass ${learner.ipass}, and #step ${learner.istep}. Required #pass > 0, or #step > [$iter] as specified")
        None
      }
    }
    var allReduceNode: Option[ActorRef] = None
    while (allReduceNode.isEmpty) {
      allReduceNode = createAllReduceNode()
      Thread.sleep(2000L)
    }
    allReduceNode.get
  }

  def startNodeAfterIter(learner: Learner, iter: Int): ActorRef = {
    val nodeConfig = getBasicConfigs()
    val binder = new ElasticAverageBinder(learner.model, nodeConfig.elasticRate)
    startNodeAfterIter(learner, iter, nodeConfig, binder)
  }

  def main(args: Array[String]): Unit = {
    val learner = new AllreduceDummyLearner()
    learner.launchTrain
    val node = startNodeAfterIter(learner, iter = 0)
    //use the following message to stop all reduce from working.
    //node ! StopAllreduceNode
  }

}

