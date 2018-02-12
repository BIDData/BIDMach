package BIDMach.allreduce

import akka.actor.{Actor, ActorRef, ActorSystem, Props, RootActorPath, Terminated}
import akka.cluster.ClusterEvent.MemberUp
import akka.cluster.{Cluster, Member}
import com.typesafe.config.ConfigFactory

import scala.collection.mutable.ArrayBuffer
import scala.concurrent.Await
import scala.concurrent.duration._


class AllreduceGridMaster(config: MasterConfig) extends Actor with akka.actor.ActorLogging {

  val nodeNum = config.nodeNum
  val addressDiscoveryTimeOut: FiniteDuration = config.discoveryTimeout

  val cluster = Cluster(context.system)

  //nodeId, nodeRef
  var nodeMap = Map[Int, ActorRef]()

  //Key: NodeIdx
  //Value: (Dimension, Array[SlavesIdx]) 
  var lineMastersAssignment = Map[Int, ArrayBuffer[(Int, ArrayBuffer[Int])]]()

  override def preStart(): Unit = cluster.subscribe(self, classOf[MemberUp])

  override def postStop(): Unit = cluster.unsubscribe(self)

  def receive = {

    case MemberUp(m) =>
      log.info(s"\n----GridMaster: Detect member ${m.address} up")
      register(m)
      if (nodeMap.size >= nodeNum) {
        println(s"----${nodeMap.size} (out of ${nodeNum}) nodes are up")
        //Step 1 
        //generate grid info and masters for each dimensions
        generateLineMasters()
        //Step 3
        //propagate the info to all the line masters who are going to be (a) line master 
        initLineMasters()
      }

    case Terminated(a) =>
      log.info(s"\n----GridMaster: $a is terminated, removing it from the map")
      for (node <- nodeMap) {
        if (node == a) {
          log.info(s"\n----GridMaster: $a should be removed. The function is NOT YET COMPLETE")
        }
      }
  }

  private def register(member: Member): Unit = {
    if (member.hasRole("Node")) {
      // awaiting here to prevent concurrent futures (from another message) trying to add to worker set at the same time
      val nodeRef: ActorRef = Await.result(context.actorSelection(RootActorPath(member.address) / "user" / "Node").resolveOne(addressDiscoveryTimeOut), addressDiscoveryTimeOut + 1.second)
      context watch nodeRef
      nodeMap = nodeMap.updated(nodeMap.size, nodeRef)
      log.info(s"\n----GridMaster: ${nodeRef} is online. Currently ${nodeMap.size} nodes are online")
    }
  }

  /*
  0 1
  2 3
  */
  private def generateLineMasters(): Unit = {
    //config 1
    // lineMastersAssignment = lineMastersAssignment.updated(0, ArrayBuffer((0, ArrayBuffer(0, 1)), (1, ArrayBuffer(0,2))))
    // lineMastersAssignment = lineMastersAssignment.updated(3, ArrayBuffer((0, ArrayBuffer(2, 3)), (1, ArrayBuffer(1,3))))

    //config 2
    //lineMastersAssignment = lineMastersAssignment.updated(0, ArrayBuffer((0, ArrayBuffer(0, 1))))
    //lineMastersAssignment = lineMastersAssignment.updated(1, ArrayBuffer((1, ArrayBuffer(1, 3))))
    //lineMastersAssignment = lineMastersAssignment.updated(2, ArrayBuffer((1, ArrayBuffer(0, 2))))
    //lineMastersAssignment = lineMastersAssignment.updated(3, ArrayBuffer((0, ArrayBuffer(2, 3))))

    //config 3
    lineMastersAssignment = lineMastersAssignment.updated(0, ArrayBuffer((0, ArrayBuffer(0, 1, 2, 3))))
    lineMastersAssignment = lineMastersAssignment.updated(5, ArrayBuffer((0, ArrayBuffer(4, 5, 6, 7))))
    lineMastersAssignment = lineMastersAssignment.updated(10, ArrayBuffer((0, ArrayBuffer(8, 9, 10, 11))))
    lineMastersAssignment = lineMastersAssignment.updated(15, ArrayBuffer((0, ArrayBuffer(12, 13, 14, 15))))
    lineMastersAssignment = lineMastersAssignment.updated(4, ArrayBuffer((1, ArrayBuffer(0, 4, 8, 12))))
    lineMastersAssignment = lineMastersAssignment.updated(9, ArrayBuffer((1, ArrayBuffer(1, 5, 9, 13))))
    lineMastersAssignment = lineMastersAssignment.updated(2, ArrayBuffer((1, ArrayBuffer(2, 6, 10, 14))))
    lineMastersAssignment = lineMastersAssignment.updated(11, ArrayBuffer((1, ArrayBuffer(3, 7, 11, 15))))

    // debug use only
    // if (nodeMap.size > nodeMap.size){
    // 	lineMastersAssignment = lineMastersAssignment.updated(0, ArrayBuffer((0, ArrayBuffer(0,1,2)), (1, ArrayBuffer(0))))
    // 	lineMastersAssignment = lineMastersAssignment.updated(2, ArrayBuffer(((1, ArrayBuffer(2)))))
    // }

  }

  private def initLineMasters(): Unit = {
    for ((nodeIdx, assignment) <- lineMastersAssignment) {
      for ((dim, slaves) <- assignment) {
        var slavenodeMapRef = ArrayBuffer[ActorRef]()
        for (slaveIdx <- slaves) {
          slavenodeMapRef += nodeMap(slaveIdx)
        }
        var lineMasterRef = discoverLineMaster(dim, nodeIdx)
        lineMasterRef ! SlavesInfo(slavenodeMapRef)
      }
    }
  }

  private def discoverLineMaster(dim: Int, masterNodeIdx: Int): ActorRef = {
    //path: masterNodePath/lineMaster-dim
    val lineMaster: ActorRef = Await.result(context.actorSelection(nodeMap(masterNodeIdx).path / s"DimensionNode-dim=${dim}" / "LineMaster")
      .resolveOne(addressDiscoveryTimeOut), addressDiscoveryTimeOut + 1.second)
    println(s"\n----GridMaster: Discover LineMaster Address : ${lineMaster.path}")
    (lineMaster)
  }

}


object AllreduceGridMaster {

  def main(args: Array[String]): Unit = {
    // Override the configuration of the port when specified as program argument
    val port = if (args.isEmpty) "2551" else args(0)
    val nodeNum = 4;
    val masterConfig = MasterConfig(nodeNum = nodeNum, discoveryTimeout = 5.seconds)

    initMaster(port, masterConfig)
  }

  private def initMaster(port: String, masterConfig: MasterConfig) = {
    val config = ConfigFactory.parseString(s"akka.remote.netty.tcp.port=$port").
      withFallback(ConfigFactory.parseString("akka.cluster.roles = [GridMaster]")).
      withFallback(ConfigFactory.load())

    val system = ActorSystem("ClusterSystem", config)

    system.log.info(s"-------\n Port = ${port} \n Number of nodes = ${masterConfig.nodeNum} \n ");
    system.actorOf(
      Props(classOf[AllreduceGridMaster], masterConfig),
      name = "GridMaster"
    )
  }

  def startUp() = {
    main(Array())
  }

  def startUp(port: String, masterConfig: MasterConfig): Unit = {
    initMaster(port, masterConfig)
  }

}



