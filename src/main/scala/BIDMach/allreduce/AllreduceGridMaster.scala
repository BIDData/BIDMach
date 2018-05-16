package BIDMach.allreduce

import akka.actor.{Actor, ActorRef, ActorSystem, Props, RootActorPath, Terminated}
import akka.cluster.ClusterEvent.MemberUp
import akka.cluster.{Cluster, Member}
import com.typesafe.config.ConfigFactory

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import scala.concurrent.Await
import scala.concurrent.duration._
import scala.util.Try


class AllreduceGridMaster(config: GridMasterConfig) extends Actor with akka.actor.ActorLogging {

  var gridLayout = new Dynamic2DGridLayout()

  val nodeNum = config.nodeNum
  val nodeResolutionTimeOut: FiniteDuration = config.nodeResolutionTimeout

  val cluster = Cluster(context.system)

  var nodesByIdMap = mutable.HashMap[Int, ActorRef]()
  var nextNodeId = 0

  // Line master version (strictly increasing) for downstream to distinguish new grid assignment
  var lineMasterVersion = -1

  var isInitialized = false

  override def preStart(): Unit = cluster.subscribe(self, classOf[MemberUp])

  override def postStop(): Unit = cluster.unsubscribe(self)

  def receive = {

    case MemberUp(joiningNode) =>
      println(s"\n----GridMaster: Detect node ${joiningNode.address} up")
      val oldLayout = gridLayout.currentMasterLayout()
      register(joiningNode)
      if (isInitialized) {
        val newLayout = gridLayout.currentMasterLayout()
        val diff = Dynamic2DGridLayout.calculate_difference(oldLayout, newLayout)
        updateAllreduceTask(diff)
      } else {
        if (nodesByIdMap.size >= nodeNum) {
          log.info(s"---- all nodes nodes are up, start allreduce")
          updateAllreduceTask(gridLayout.currentMasterLayout())
          isInitialized = true
        }
      }
    case Terminated(a) =>
      log.info(s"\n----GridMaster: $a is terminated, removing it from the grid")
      val oldLayout = gridLayout.currentMasterLayout()
      for ((nodeId, nodeRef) <- nodesByIdMap) {
        if (nodeRef == a) {
          gridLayout.removeNode(nodeId)
          nodesByIdMap.remove(nodeId)
        }
      }
      if (isInitialized) {
        val newLayout = gridLayout.currentMasterLayout()
        val diff = Dynamic2DGridLayout.calculate_difference(oldLayout, newLayout)
        updateAllreduceTask(diff)
      }
  }

  /**
    * Register a node and maintain related information
    * @param node the node to be added
    */
  private def register(node: Member): Unit = {
    if (node.hasRole("Node")) {
      // awaiting here to prevent concurrent futures (from another message) trying to add to node set at the same time
      val result: Try[ActorRef] = Await.ready(context.actorSelection(RootActorPath(node.address) / "user" / "Node")
        .resolveOne(nodeResolutionTimeOut), nodeResolutionTimeOut + 1.second).value.get
      if (result.isSuccess) {
        val nodeRef = result.get
        context watch nodeRef
        nodesByIdMap(nextNodeId) = nodeRef
        gridLayout.addNode(nextNodeId)
        nextNodeId += 1
        println(s"\n----GridMaster: ${nodeRef} is online. Currently ${nodesByIdMap.size} nodes are online")
      }else{
        println(s"\n----GridMaster: Fail to discover Node ${node} Address")
      }
    }
    println("Done with register")
  }

  /**
    * Send to each linemaster about the update via difference from Dynamic 2D Grid
    *
    * @param diff difference calculated by Dynamic 2D Grid
    */
  private def updateAllreduceTask(diff: Dynamic2DGridLayout.MasterLayout): Unit = {
    log.info(s"\n updating Layout: ${diff}")
    lineMasterVersion += 1
    for ((lineMasterNodeId, assignments) <- diff if nodesByIdMap.contains(lineMasterNodeId)) {
      for ((iter, dim) <- assignments.productIterator.zipWithIndex) {
        // here we only deal with it when both assigment and lineMaster exists
        for (assignment <- iter.asInstanceOf[Option[Set[Int]]]; lineMasterRef <- discoverLineMaster(dim, lineMasterNodeId)) {
          if (assignment.isEmpty) {
            lineMasterRef ! StopAllreduceTask(lineMasterVersion)
          } else {
            var peerNodeRefs = ArrayBuffer[ActorRef]()
            for (nodeId <- assignment) {
              peerNodeRefs += nodesByIdMap(nodeId)
            }
            println(s"Start all reduce task to $lineMasterRef")
            lineMasterRef ! StartAllreduceTask(peerNodeRefs, lineMasterVersion)
          }
        }
      }
    }
    log.info(s"\n update complete")
  }

  private def discoverLineMaster(dim: Int, masterNodeIdx: Int): Option[ActorRef] = {
    val result: Try[ActorRef] = Await.ready(context.actorSelection(nodesByIdMap(masterNodeIdx).path / s"DimensionNode-dim=${dim}" / "LineMaster")
      .resolveOne(nodeResolutionTimeOut), nodeResolutionTimeOut + 1.second).value.get
    if (result.isSuccess) {
      log.info(s"\n----GridMaster: Discover LineMaster ${masterNodeIdx}-${dim} Address to be ${result.get.path}")
      Some(result.get)
    } else {
      log.info(s"\n----GridMaster: Fail to discover LineMaster ${masterNodeIdx}-${dim} Address")
      Option.empty
    }

  }

}


object AllreduceGridMaster {

  def main(args: Array[String]): Unit = {
    // Override the configuration of the port when specified as program argument
    val port = if (args.isEmpty) "2551" else args(0)
    val nodeNum = 1
    val masterConfig = GridMasterConfig(nodeNum = nodeNum, nodeResolutionTimeout = 5.seconds)

    initGridMaster(port, masterConfig)
  }

  private def initGridMaster(port: String, gridMasterConfig: GridMasterConfig) = {
    val config = ConfigFactory.parseString(s"akka.remote.netty.tcp.port=$port").
      withFallback(ConfigFactory.parseString("akka.cluster.roles = [GridMaster]")).
      withFallback(ConfigFactory.load())

    val system = ActorSystem("ClusterSystem", config)

    system.log.info(s"-------\n Port = ${port} \n Number of nodes = ${gridMasterConfig.nodeNum} \n ");
    system.actorOf(
      Props(classOf[AllreduceGridMaster], gridMasterConfig),
      name = "GridMaster"
    )
  }

  def startUp() = {
    main(Array())
  }

  def startUp(port: String, masterConfig: GridMasterConfig): Unit = {
    initGridMaster(port, masterConfig)
  }

}

case class LineAssignment(masterNodeId: Int, dimension: Int, peerNodeIds: ArrayBuffer[Int])

