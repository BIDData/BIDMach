package BIDMach.allreduce

import akka.actor.{Actor, ActorRef, ActorSystem, Props, RootActorPath, Terminated}
import akka.cluster.ClusterEvent.MemberUp
import akka.cluster.{Cluster, Member}
import com.typesafe.config.ConfigFactory

import scala.collection.mutable.ArrayBuffer
import scala.concurrent.Await
import scala.concurrent.duration._


class AllreduceGridMaster(config: GridMasterConfig) extends Actor with akka.actor.ActorLogging {

  val nodeNum = config.nodeNum
  val nodeResolutionTimeOut: FiniteDuration = config.nodeResolutionTimeout

  val cluster = Cluster(context.system)

  var nodeReferenceByIdMap = Map[Int, ActorRef]()

  // Specifies line assignment for a line master denoted by its node id
  var lineMastersAssignment = Map[Int, ArrayBuffer[LineAssignment]]()

  // Line master version (strictly increasing) for downstream to distinguish new grid assignment
  var lineMasterVersion = -1

  override def preStart(): Unit = cluster.subscribe(self, classOf[MemberUp])

  override def postStop(): Unit = cluster.unsubscribe(self)

  def receive = {

    case MemberUp(m) =>
      log.info(s"\n----GridMaster: Detect node ${m.address} up")
      register(m)
      if (nodeReferenceByIdMap.size >= nodeNum) {
        println(s"----${nodeReferenceByIdMap.size} (out of ${nodeNum}) nodes are up")
        //Step 1
        //generate grid info and masters for each dimensions
        generateGridAssignment(nodeNum)
        //Step 2
        //propagate the info to all the line masters who are going to be (a) line master
        startAllreduceTask()
      }

    case Terminated(a) =>
      log.info(s"\n----GridMaster: $a is terminated, removing it from the map")
      for ((_, nodeRef) <- nodeReferenceByIdMap) {
        if (nodeRef == a) {
          log.info(s"\n----GridMaster: $a should be removed. The function is NOT YET COMPLETE")
        }
      }
  }

  private def register(node: Member): Unit = {
    if (node.hasRole("Node")) {
      // awaiting here to prevent concurrent futures (from another message) trying to add to node set at the same time
      val nodeRef: ActorRef = Await.result(context.actorSelection(RootActorPath(node.address) / "user" / "Node").resolveOne(nodeResolutionTimeOut), nodeResolutionTimeOut + 1.second)
      context watch nodeRef
      nodeReferenceByIdMap = nodeReferenceByIdMap.updated(nodeReferenceByIdMap.size, nodeRef)
      log.info(s"\n----GridMaster: ${nodeRef} is online. Currently ${nodeReferenceByIdMap.size} nodes are online")
    }
  }

  private def generateGridAssignment(nodeNum: Int): Unit = {

    if (nodeNum == 4) {
      lineMastersAssignment = lineMastersAssignment.updated(0, ArrayBuffer(LineAssignment(0, ArrayBuffer(0, 1))))
      lineMastersAssignment = lineMastersAssignment.updated(1, ArrayBuffer(LineAssignment(1, ArrayBuffer(1, 3))))
      lineMastersAssignment = lineMastersAssignment.updated(2, ArrayBuffer(LineAssignment(1, ArrayBuffer(0, 2))))
      lineMastersAssignment = lineMastersAssignment.updated(3, ArrayBuffer(LineAssignment(0, ArrayBuffer(2, 3))))
    } else if (nodeNum == 16) {
      lineMastersAssignment = lineMastersAssignment.updated(0, ArrayBuffer(LineAssignment(0, ArrayBuffer(0, 1, 2, 3))))
      lineMastersAssignment = lineMastersAssignment.updated(5, ArrayBuffer(LineAssignment(0, ArrayBuffer(4, 5, 6, 7))))
      lineMastersAssignment = lineMastersAssignment.updated(10, ArrayBuffer(LineAssignment(0, ArrayBuffer(8, 9, 10, 11))))
      lineMastersAssignment = lineMastersAssignment.updated(15, ArrayBuffer(LineAssignment(0, ArrayBuffer(12, 13, 14, 15))))
      lineMastersAssignment = lineMastersAssignment.updated(4, ArrayBuffer(LineAssignment(1, ArrayBuffer(0, 4, 8, 12))))
      lineMastersAssignment = lineMastersAssignment.updated(9, ArrayBuffer(LineAssignment(1, ArrayBuffer(1, 5, 9, 13))))
      lineMastersAssignment = lineMastersAssignment.updated(2, ArrayBuffer(LineAssignment(1, ArrayBuffer(2, 6, 10, 14))))
      lineMastersAssignment = lineMastersAssignment.updated(11, ArrayBuffer(LineAssignment(1, ArrayBuffer(3, 7, 11, 15))))
    } else {
      throw new IllegalArgumentException(s"Hard-coded line master only support 4 and 16 nodes, but given node number is $nodeNum")
    }

    // debug use only
    // if (nodeMap.size > nodeMap.size){
    // lineMastersAssignment = lineMastersAssignment.updated(0, ArrayBuffer(LineOrganization(0, ArrayBuffer(0,1,2)), LineOrganization(1, ArrayBuffer(0))))
    // lineMastersAssignment = lineMastersAssignment.updated(2, ArrayBuffer((LineOrganization(1, ArrayBuffer(2)))))
    // }

  }

  private def startAllreduceTask(): Unit = {
    lineMasterVersion += 1
    for ((lineMasterNodeId, assignments) <- lineMastersAssignment) {
      for (assignment <- assignments) {
        val peerNodeIds = assignment.peerNodeIds
        val dim = assignment.dimension
        var peerNodeRefs = ArrayBuffer[ActorRef]()
        for (nodeId <- peerNodeIds) {
          peerNodeRefs += nodeReferenceByIdMap(nodeId)
        }
        val lineMasterRef = discoverLineMaster(dim, lineMasterNodeId)
        lineMasterRef ! StartAllreduceTask(peerNodeRefs, lineMasterVersion)
      }
    }
  }

  private def discoverLineMaster(dim: Int, masterNodeIdx: Int): ActorRef = {
    //path: masterNodePath/lineMaster-dim

    val lineMaster: ActorRef = Await.result(context.actorSelection(nodeReferenceByIdMap(masterNodeIdx).path / s"DimensionNode-dim=${dim}" / "LineMaster")
      .resolveOne(nodeResolutionTimeOut), nodeResolutionTimeOut + 1.second)
    println(s"\n----GridMaster: Discover LineMaster Address : ${lineMaster.path}")
    (lineMaster)
  }

}


object AllreduceGridMaster {

  def main(args: Array[String]): Unit = {
    // Override the configuration of the port when specified as program argument
    val port = if (args.isEmpty) "2551" else args(0)
    val nodeNum = 4;
    val masterConfig = GridMasterConfig(nodeNum = nodeNum, nodeResolutionTimeout = 5.seconds)

    initMaster(port, masterConfig)
  }

  private def initMaster(port: String, masterConfig: GridMasterConfig) = {
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

  def startUp(port: String, masterConfig: GridMasterConfig): Unit = {
    initMaster(port, masterConfig)
  }

}

case class LineAssignment(dimension: Int, peerNodeIds: ArrayBuffer[Int])

