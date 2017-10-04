package BIDMach.allreduce

import java.util.concurrent.atomic.AtomicInteger

import akka.Done
import akka.actor.{Actor, ActorRef, ActorSystem, Props, RootActorPath, Terminated}
import akka.cluster.ClusterEvent.{CurrentClusterState, MemberUp}
import akka.cluster.{Cluster, Member, MemberStatus}
import akka.pattern.ask
import akka.util.Timeout
import com.typesafe.config.ConfigFactory

import scala.concurrent.ExecutionContext.Implicits.global
import scala.concurrent.duration._
import scala.language.postfixOps
import scala.util.{Failure, Success}

class GridMaster extends Actor {

  var workers: Set[ActorRef] = Set.empty[ActorRef]

  val cluster = Cluster(context.system)

  override def preStart(): Unit = cluster.subscribe(self, classOf[MemberUp])

  override def postStop(): Unit = cluster.unsubscribe(self)

  def receive = {

    // Organize grid
    case command: OrganizeGridWorker =>
      if (workers.isEmpty) {
        sender() ! GridOrganizationFailed("No workers available, try again later", command)
      } else {
        workers foreach { each =>
          each ! GridNeighborAddresses(workers.toSeq)
        }
        sender() ! Done
      }

    // Cluster management
    case state: CurrentClusterState =>
      println(s"Current state $state")
      state.members.filter(_.status != MemberStatus.Up) foreach register

    case MemberUp(m) => register(m)

    case Terminated(a) =>
      println(s"$a is terminated, removing it from the set")
      workers = workers.filterNot(_ == a)

  }

  def register(member: Member): Unit =
    if (member.hasRole("worker")) {
      implicit val timeout = Timeout(5.seconds)
      context.actorSelection(RootActorPath(member.address) / "user" / "worker").resolveOne().onComplete {
        case Success(workerRef: ActorRef) =>
          context watch workerRef
          workers = workers + workerRef
      }
    }

}


object GridMaster {
  def main(args: Array[String]): Unit = {
    // Override the configuration of the port when specified as program argument
    val port = if (args.isEmpty) "0" else args(0)
    val config = ConfigFactory.parseString(s"akka.remote.netty.tcp.port=$port").
      withFallback(ConfigFactory.parseString("akka.cluster.roles = [master]")).
      withFallback(ConfigFactory.load())

    val system = ActorSystem("ClusterSystem", config)
    val master = system.actorOf(Props[GridMaster], name = "master")


    // To constantly try to organize grid workers
    val counter = new AtomicInteger
    system.scheduler.schedule(2.seconds, 2.seconds) {
      implicit val timeout = Timeout(5 seconds)
      (master ? OrganizeGridWorker("counter-" + counter.incrementAndGet())) onComplete {
        case Success(result) => println(result)
        case Failure(e) => println(s"Error $e")
      }
    }

  }
}