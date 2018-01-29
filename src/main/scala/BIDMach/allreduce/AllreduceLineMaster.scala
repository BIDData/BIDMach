package BIDMach.allreduce

import akka.actor.{Actor, ActorRef, ActorSystem, Props, RootActorPath, Terminated}
import akka.cluster.ClusterEvent.MemberUp
import akka.cluster.{Cluster, Member}
import com.typesafe.config.ConfigFactory

import scala.concurrent.{Await, Future}
import scala.concurrent.duration._
import scala.language.postfixOps
import scala.concurrent.ExecutionContext.Implicits.global
import scala.collection.mutable.ArrayBuffer



class AllreduceLineMaster(config: LineMasterConfig) extends Actor with akka.actor.ActorLogging{

  var gridMaster : Option[ActorRef] = None
  var workerNum = -1
  var roundNum = config.workerPerNodeNum //the number of rounds (lags) allowed
  var dim = config.dim

  val thAllreduce = config.threshold.thAllreduce
  val maxRound = config.maxRound

  val addressDiscoveryTimeOut: FiniteDuration = config.discoveryTimeout

  var round = 0
  var workerMap: Map[Int, ActorRef] = Map() // workers in the same row/col, including self
  var nodeMap: Map[Int, ActorRef] = Map()
  var completeCount = 0
  var confirmPrepareCount = 0

  def receive = {

    case confirm: ConfirmPreparation => {
      //log.info(s"\n----LineMaster ${self.path} receive confimation from ${sender} with round ${confirm.round}");
      if (confirm.round == round) {
        confirmPrepareCount += 1
        //log.info(s"\n----LineMaster ${self.path} receive confimation from ${sender}; ${confirmPrepareCount} out of ${workerNum}")
        if (confirmPrepareCount == workerNum) {
          startAllreduce()
        }
      }
    }

    case c : CompleteAllreduce =>
      //log.info(s"\n----LineMaster ${self.path}: Node ${c.srcId} completes allreduce round ${c.round}")
      if (c.round == round) {
        completeCount += 1
        if (completeCount >= workerNum * thAllreduce && round < maxRound) {
          //log.info(s"\n----LineMaster ${self.path}: ${completeCount} (out of ${workerNum}) workers complete round ${round}\n")
          round += 1
          prepareAllreduce()
        }
      }

    case slavesInfo: SlavesInfo =>
      log.info(s"\n----LineMaster ${self.path}: Receive SlavesInfo from GridMaster.")
      gridMaster = Some(sender())
      nodeMap = slavesInfo.slaveNodesRef.zipWithIndex.map(tup => (tup._2, tup._1)).toMap
      //if the LM hasnt begun, initiate PrepareAllreduce. 
      //Otherwise, we just update the nodeMap and wait for the current round to end
      if (round == 0){
        prepareAllreduce()
      }
    

  }

  private def startAllreduce() = {
    log.info(s"\n----LineMaster ${self.path}: START ROUND ${round} at time ${System.currentTimeMillis} --------------------")
    completeCount = 0
    for (worker <- workerMap.values) {
      worker ! StartAllreduce(round)
    }
  }

  private def prepareAllreduce() = {
    //log.info(s"\n----LineMaster ${self.path}: Preparing allreduce round ${round}")
    confirmPrepareCount = 0
    workerNum = nodeMap.size
    workerMap = discoverWorkers(round, nodeMap)
    for ((nodeIndex, worker) <- workerMap) {
      //log.info(s"\n----LineMaster ${self.path}: Sending prepare msg to worker $worker")
      worker ! PrepareAllreduce(round, workerMap, nodeIndex)
    }
  }

  private def discoverWorkers(round: Int, nodeMap: Map[Int, ActorRef]): Map[Int, ActorRef] = {
    val addressesFut: Seq[Future[(Int, ActorRef)]] = nodeMap.toSeq.map {
      case (nodeId, nodeAddress) =>
        //nodePath/worker-id-dim
        context.actorSelection(nodeAddress.path / s"DimensionNode-dim=${dim}" / s"Worker-id=${round % roundNum}")
          .resolveOne(addressDiscoveryTimeOut)
            .map(ref => (nodeId, ref))

    }
    Await.result(Future.sequence(addressesFut), addressDiscoveryTimeOut).toMap
  }
}

object AllreduceLineMaster {

  def main(args: Array[String]): Unit = {

    val workerPerNodeNum = 3
    val dataSize = 500000

    val maxChunkSize = 20000

    val maxRound = 3000

    val threshold = ThresholdConfig(thAllreduce = 0.5f, thReduce = 0.5f, thComplete = 0.5f)
    val metaData = MetaDataConfig(dataSize = dataSize, maxChunkSize = maxChunkSize)
    val masterConfig = LineMasterConfig(workerPerNodeNum = workerPerNodeNum, dim=0, maxRound,
      discoveryTimeout = 5.seconds,
      threshold = threshold,
      metaData= metaData)

    AllreduceLineMaster.startUp("2551", threshold, metaData, masterConfig)
  }

  private def initMaster(port: String, lineMasterConfig: LineMasterConfig) = {
    val config = ConfigFactory.parseString(s"\nakka.remote.netty.tcp.port=$port").
      withFallback(ConfigFactory.parseString("akka.cluster.roles = [master]")).
      withFallback(ConfigFactory.load())
    val system = ActorSystem("ClusterSystem", config)


    system.log.info(s"-------\n Port = ${port} \n Message Size = ${lineMasterConfig.metaData.dataSize} \n Max Chunk Size = ${lineMasterConfig.metaData.maxChunkSize}");
    system.actorOf(
      Props(classOf[AllreduceLineMaster], lineMasterConfig),
      name = "master"
    )
  }

  def startUp(port: String, thresholds: ThresholdConfig, dataConfig: MetaDataConfig, lineMasterConfig: LineMasterConfig): Unit = {
    initMaster(port, lineMasterConfig)
  }

}
