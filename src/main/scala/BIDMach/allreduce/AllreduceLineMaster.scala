package BIDMach.allreduce

import akka.actor.{Actor, ActorRef, ActorSystem, Props, RootActorPath, Terminated}
import akka.cluster.ClusterEvent.MemberUp
import akka.cluster.{Cluster, Member}
import com.typesafe.config.ConfigFactory

import scala.concurrent.{Await, Future}
import scala.concurrent.duration._
import scala.language.postfixOps
import scala.concurrent.ExecutionContext.Implicits.global

class AllreduceLineMaster(config: MasterConfig) extends Actor with akka.actor.ActorLogging{

  val nodeNum = config.nodeNum
  val workerNum = config.workerNum

  val thAllreduce = config.threshold.thAllreduce
  val maxRound = config.maxRound

  val cluster = Cluster(context.system)
  val addressDiscoveryTimeOut: FiniteDuration = config.discoveryTimeout

  var nodes = Set[ActorRef]()


  var round = -1
  var workerMap: Map[Int, ActorRef] = Map() // workers in the same row/col, including self
  var completeCount = 0
  var confirmPrepareCount = 0


  override def preStart(): Unit = cluster.subscribe(self, classOf[MemberUp])

  override def postStop(): Unit = cluster.unsubscribe(self)

  def receive = {

    case MemberUp(m) =>
      log.info(s"\n----Detect member ${m.address} up")
      register(m)
      if (nodes.size >= nodeNum && round == -1) {
        println(s"----${nodes.size} (out of ${nodeNum}) workers are up")
        round = 0
        prepareAllreduce()
      }

    case Terminated(a) =>
      log.info(s"\n----$a is terminated, removing it from the set")
      for (node <- nodes){
        if(node == a) {
          nodes -= node
        }
      }

    case confirm: ConfirmPreparation => {
      if (confirm.round == round) {
        confirmPrepareCount += 1
        if (confirmPrepareCount == nodeNum) {
          startAllreduce()
        }
      }
    }

    case c : CompleteAllreduce =>
      log.debug(s"\n----Node ${c.srcId} completes allreduce round ${c.round}")
      if (c.round == round) {
        completeCount += 1
        if (completeCount >= nodeNum * thAllreduce && round < maxRound) {
          log.info(s"----${completeCount} (out of ${nodeNum}) workers complete round ${round}\n")
          round += 1
          prepareAllreduce()
        }
      }
  }

  private def register(member: Member): Unit =
    if (member.hasRole("node")) {
      // awaiting here to prevent concurrent futures (from another message) trying to add to worker set at the same time
      val workerRef: ActorRef = Await.result(context.actorSelection(RootActorPath(member.address) / "user" / "node").resolveOne(addressDiscoveryTimeOut), addressDiscoveryTimeOut + 1.second)
      context watch workerRef
      nodes += workerRef
      log.info(s"\n----current size = ${nodes.size}")
    }

  private def startAllreduce() = {
    println(s"\n----Start allreduce round ${round}")
    completeCount = 0
    for (worker <- workerMap.values) {
      worker ! StartAllreduce(round)
    }
  }

  private def prepareAllreduce() = {
    println(s"\n----Preparing allreduce round ${round}")
    confirmPrepareCount = 0

    val nodeMap: Map[Int, ActorRef] = nodes.zipWithIndex.map(tup => (tup._2, tup._1)).toMap
    workerMap = discoverWorkers(round, nodeMap)
    for ((nodeIndex, worker) <- workerMap) {
      println(s"\n----Sending prepare msg to worker $worker")
      worker ! PrepareAllreduce(round, nodeMap, nodeIndex)
    }
  }

  private def discoverWorkers(round: Int, nodeMap: Map[Int, ActorRef]): Map[Int, ActorRef] = {
    val addressesFut: Seq[Future[(Int, ActorRef)]] = nodeMap.toSeq.map {
      case (nodeId, nodeAddress) =>
        context.actorSelection(nodeAddress.path / s"worker-${round % workerNum}")
          .resolveOne(addressDiscoveryTimeOut)
            .map(ref => (nodeId, ref))

    }
    Await.result(Future.sequence(addressesFut), addressDiscoveryTimeOut).toMap
  }
}



object AllreduceLineMaster {

  def main(args: Array[String]): Unit = {
    // Override the configuration of the port when specified as program argument

    val workerNum = 1
    val maxRound = 100

    val port = if (args.isEmpty) "2551" else args(0)
    val nodeNum = if (args.length <= 1) 2 else args(1).toInt
    val dataSize = if (args.length <= 2) nodeNum * 5 else args(2).toInt
    val maxChunkSize = if (args.length <= 3) 2 else args(3).toInt


    val threshold = ThresholdConfig(thAllreduce = 1f, thReduce = 1f, thComplete = 0.8f)
    val metaData = MetaDataConfig(dataSize = dataSize, maxChunkSize = maxChunkSize)
    val masterConfig = MasterConfig(nodeNum = nodeNum, workerNum = workerNum, maxRound,
      discoveryTimeout = 5.seconds,
      threshold = threshold,
      metaData= metaData)

    initMaster(port, masterConfig)
  }

  private def initMaster(port: String, masterConfig: MasterConfig) = {
    val config = ConfigFactory.parseString(s"\nakka.remote.netty.tcp.port=$port").
      withFallback(ConfigFactory.parseString("akka.cluster.roles = [master]")).
      withFallback(ConfigFactory.load())

    val system = ActorSystem("ClusterSystem", config)


    system.log.info(s"-------\n Port = ${port} \n Number of nodes = ${masterConfig.nodeNum} \n Message Size = ${masterConfig.metaData.dataSize} \n Max Chunk Size = ${masterConfig.metaData.maxChunkSize}");
    system.actorOf(
      Props(classOf[AllreduceLineMaster], masterConfig),
      name = "master"
    )
  }

  def startUp() = {
    main(Array())
  }

  def startUp(port: String, thresholds: ThresholdConfig, dataConfig: MetaDataConfig, masterConfig: MasterConfig): Unit = {

    initMaster(port, masterConfig)
  }

}

case class ThresholdConfig(thAllreduce: Float, thReduce: Float, thComplete: Float)
case class MetaDataConfig(dataSize: Int, maxChunkSize: Int)
case class WorkerConfig(workerNum: Int,
                        discoveryTimeout: FiniteDuration,
                        threshold: ThresholdConfig,
                        metaData: MetaDataConfig)
case class MasterConfig(nodeNum: Int, workerNum: Int, maxRound: Int,
                        discoveryTimeout: FiniteDuration,
                        threshold: ThresholdConfig,
                        metaData: MetaDataConfig)

