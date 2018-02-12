package BIDMach.allreduce

import akka.actor.{Actor, ActorRef}

import scala.concurrent.ExecutionContext.Implicits.global
import scala.concurrent.duration._
import scala.concurrent.{Await, Future}
import scala.language.postfixOps


class AllreduceLineMaster(config: LineMasterConfig) extends Actor with akka.actor.ActorLogging {

  var gridMaster: Option[ActorRef] = None
  var workerNum = -1
  var roundNum = config.workerPerNodeNum //the number of rounds (lags) allowed
  var dim = config.dim

  val thAllreduce = config.threshold.thAllreduce
  val maxRound = config.maxRound

  val addressDiscoveryTimeOut: FiniteDuration = config.discoveryTimeout

  var lineMasterVersion = -1
  var round = -1

  // worker address for all rounds
  var workerMapAcrossRounds: Array[Map[Int, ActorRef]] = new Array(roundNum)

  var completeCount = 0

  def receive = {

    case c: CompleteAllreduce =>
      log.debug(s"\n----LineMaster ${self.path}: Node ${c.srcId} completes allreduce round ${c.round}")
      if (c.config.round == round) {
        completeCount += 1
        if (completeCount >= workerNum * thAllreduce && round < maxRound) {
          //log.info(s"\n----LineMaster ${self.path}: ${completeCount} (out of ${workerNum}) workers complete round ${round}\n")
          round += 1
          startAllreduce()
        }
      }

    case s: StartAllreduceTask =>
      log.debug(s"\n----LineMaster ${self.path}: Receive SlavesInfo from GridMaster.")
      gridMaster = Some(sender())
      lineMasterVersion = s.lineMasterVersion
      val nodeRefs = s.slaveNodesRef
      workerNum = nodeRefs.size
      for (workerRound <- 0 until roundNum) {
        workerMapAcrossRounds(workerRound) = discoverWorkers(workerRound, nodeRefs.toArray)
      }
      round += 1
      startAllreduce()
  }

  private def startAllreduce() = {
    log.debug(s"\n----LineMaster ${self.path}: START ROUND ${round} at time ${System.currentTimeMillis} --------------------")
    completeCount = 0
    val roundWorkerMap = workerMapAcrossRounds(timeIdx(round))
    for ((nodeIndex, worker) <- roundWorkerMap) {
      worker ! StartAllreduce(RoundConfig(lineMasterVersion, round, self, roundWorkerMap, nodeIndex))
    }
  }

  private def timeIdx(round: Int) = {
    round % roundNum
  }

  private def discoverWorkers(round: Int, nodeArray: Array[ActorRef]): Map[Int, ActorRef] = {
    val addressesFut: Seq[Future[(Int, ActorRef)]] = nodeArray.zipWithIndex.map {
      case (nodeAddress, nodeId) =>

        //nodePath/worker-id-dim
        context.actorSelection(nodeAddress.path / s"DimensionNode-dim=${dim}" / s"Worker-id=${round % roundNum}")
          .resolveOne(addressDiscoveryTimeOut)
          .map(ref => (nodeId, ref))

    }
    Await.result(Future.sequence(addressesFut), addressDiscoveryTimeOut).toMap
  }
}