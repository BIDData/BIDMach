package BIDMach.allreduce

import akka.Done
import akka.actor.{Actor, ActorRef, ActorSystem, Props, RootActorPath, Terminated}
import akka.cluster.ClusterEvent.{CurrentClusterState, MemberUp}
import akka.cluster.{Cluster, Member, MemberStatus}
import akka.util.Timeout
import com.typesafe.config.ConfigFactory

import scala.concurrent.ExecutionContext.Implicits.global
import scala.concurrent.Future
import scala.concurrent.duration._
import scala.language.postfixOps

class GridMaster(layout: GridLayout) extends Actor {

  var workers: collection.mutable.Map[Integer, ActorRef] = collection.mutable.Map[Integer, ActorRef]()
  var broadcasted: Boolean = false

  val cluster = Cluster(context.system)
  println(s"current layout: ${layout.scale}, ${layout.dim}")

  override def preStart(): Unit = cluster.subscribe(self, classOf[MemberUp])

  override def postStop(): Unit = cluster.unsubscribe(self)

  def receive = {

    // Cluster management
    case state: CurrentClusterState =>
      println(s"Current state $state")
      state.members.filter(_.status != MemberStatus.Up) foreach register

    case MemberUp(m) =>
      if(!broadcasted) {
        println(s"detect member up, currently ${workers.size}")
        register(m).onSuccess {
          case Done =>
            if (workers.size == layout.total) {
              println(s"expected members up, start broadcasting layout")
              setupLayout()
              broadcasted = true
            }
        }
      }

    case Terminated(a) =>
      println(s"$a is terminated, removing it from the set")
      for ((idx, worker) <- workers){
        if(worker == a) workers -= idx
      }
  }

  def setupLayout(): Unit =
  // Organize grid
  for((idx, worker) <- workers){
    val groups = layout.groups(idx)
    for (group <- groups){
      val member_idxs = layout.members(group)
      var members = Set[ActorRef]()
      for(member_id <- member_idxs) members+=workers(member_id)
      val addresses = GridGroupAddresses(group, members)
      println(s"To worker $idx $worker: Sending group address: $addresses")
      worker ! addresses
    }
  }

  def register(member: Member): Future[Done] =
    if (member.hasRole("worker")) {
      implicit val timeout = Timeout(5.seconds)
      context.actorSelection(RootActorPath(member.address) / "user" / "worker").resolveOne().map { workerRef =>
          context watch workerRef
          val new_idx: Integer = workers.size
          workers.update(new_idx, workerRef)
          Done
      }
    } else {
      Future.successful(Done)
    }

}


object GridMaster {

  def main(args: Array[String]): Unit = {
    // Override the configuration of the port when specified as program argument
    val port = if (args.isEmpty) "0" else args(0)
    val config = ConfigFactory.parseString(s"akka.remote.netty.tcp.port=$port").
      withFallback(ConfigFactory.parseString("akka.cluster.roles = [master]")).
      withFallback(ConfigFactory.load())

    val layout: GridLayout = new GridLayout(2,2); // current fixed layout for four machines

    val system = ActorSystem("ClusterSystem", config)
    val master = system.actorOf(Props(classOf[GridMaster],layout), name = "master")

  }

  def startUp(ports: List[String] = List("2551")): Unit = {
    ports foreach( eachPort => main(Array(eachPort)))
  }
}