package BIDMach.allreduce

import BIDMach.Learner
import BIDMach.allreduce.binder.AllreduceBinder
import BIDMat.Mat
import akka.actor.{ActorRef, ActorSystem, Props}
import com.typesafe.config.ConfigFactory


class AllreduceLayer(metaData: MetaDataConfig,
                     nodeConfig: NodeConfig,
                     workerConfig: WorkerConfig,
                     lineMasterConfig: LineMasterConfig) {

  var started = false

  def start(binder: AllreduceBinder): ActorRef = {

    if (started) {
      throw new IllegalStateException(s"Actor system has already started, and this node has joined the cluster, and cannot be started again. Consider restarting the actor system")
    }

    val metaDataWithSize = metaData.copy(dataSize = binder.totalDataSize)
    val config = ConfigFactory.parseString(s"akka.remote.netty.tcp.port=0").
      withFallback(ConfigFactory.parseString("akka.cluster.roles = [Node]")).
      withFallback(ConfigFactory.load())

    val system = ActorSystem("ClusterSystem", config)
    val nodeRef = system.actorOf(Props(classOf[AllreduceNode],
      nodeConfig,
      lineMasterConfig.copy(metaData = metaDataWithSize),
      workerConfig.copy(metaData = metaDataWithSize),
      binder
    ), name = "Node")

    started = true
    nodeRef
  }

  def startAfterIter(learner: Learner, iter: Int)(binderProvider: Array[Mat] => AllreduceBinder): ActorRef = {

    def createAllReduceNode(): Option[ActorRef] = {
      if (learner.synchronized(learner.ipass > 0 || learner.istep > iter)) {
        val allReduceNode = start(binderProvider(learner.modelmats))
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


}