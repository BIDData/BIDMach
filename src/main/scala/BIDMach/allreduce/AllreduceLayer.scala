package BIDMach.allreduce

import java.util.concurrent.TimeoutException

import BIDMach.Learner
import akka.actor.{ActorRef, ActorSystem, Props}
import akka.pattern.Patterns.after

import scala.concurrent.ExecutionContext.Implicits.global
import scala.concurrent.Future
import scala.concurrent.duration._


class AllreduceLayer(actorSystem: ActorSystem,
                     threshold: ThresholdConfig,
                     metaData: MetaDataConfig,
                     nodeConfig: NodeConfig,
                     workerConfig: WorkerConfig,
                     lineMasterConfig: LineMasterConfig) {


  def startAfterIter(learner: Learner, iter: Int) = {

    def createAllReduceNode(): Future[ActorRef] = {
      if (learner.synchronized(learner.ipass > iter || learner.istep > iter)) {
        val binder = new AllreduceBinder(learner.model.modelmats)
        val metaDataWithSize = metaData.copy(dataSize = binder.totalLength)
        val allReduceNode = actorSystem.actorOf(Props(classOf[AllreduceNode],
          nodeConfig,
          lineMasterConfig.copy(metaData = metaDataWithSize),
          workerConfig.copy(metaData = metaDataWithSize),
          binder
        ), name = "Node")
        println("All reduce node created.")
        Future.successful(allReduceNode)
      } else {
        Future.failed(new TimeoutException("Learner hasn't proceeded"))
      }
    }

    def createAllReduceNodeWithRetry(): Future[ActorRef] = {
      createAllReduceNode().recoverWith {
        case _: TimeoutException => after(2.seconds, actorSystem.scheduler, global, createAllReduceNodeWithRetry())
        case ex: Exception => throw ex
      }
    }

    createAllReduceNodeWithRetry()

  }


}