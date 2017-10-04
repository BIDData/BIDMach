package sample.cluster.grid

import java.util.concurrent.atomic.AtomicInteger

import akka.actor.{Actor, ActorRef, ActorSystem, Props, Terminated}
import akka.pattern.ask
import akka.util.Timeout
import com.typesafe.config.ConfigFactory

import scala.concurrent.ExecutionContext.Implicits.global
import scala.concurrent.duration._
import scala.language.postfixOps

class GridWorker extends Actor {

  var myNeighbors: Set[ActorRef] = Set.empty[ActorRef]
  val counter = new AtomicInteger

  def receive = {

    // Do work
    case GreetNeighbor =>
      myNeighbors foreach { each =>
        each ! HelloFromNeighbor(s"greeting! ${counter.getAndIncrement()}")
      }

    case HelloFromNeighbor(msg) =>
      println(s"Receiving greeting from ${sender()} msg: $msg")

    // Manage neighbors
    case GridNeighborAddresses(list) =>
      list foreach register

    case Terminated(a) =>
      println(s"Neighbor $a is terminated, removing it from the set")
      myNeighbors = myNeighbors - a

  }

  def register(neighborRef: ActorRef): Unit = {
    if (!myNeighbors.contains(neighborRef) && neighborRef != self) {
      context watch neighborRef
      myNeighbors = myNeighbors + neighborRef
    }
  }

}

object GridWorker {
  def main(args: Array[String]): Unit = {
    // Override the configuration of the port when specified as program argument
    val port = if (args.isEmpty) "0" else args(0)
    val config = ConfigFactory.parseString(s"akka.remote.netty.tcp.port=$port").
      withFallback(ConfigFactory.parseString("akka.cluster.roles = [worker]")).
      withFallback(ConfigFactory.load())

    val system = ActorSystem("ClusterSystem", config)
    val backend = system.actorOf(Props[GridWorker], name = "worker")

    system.scheduler.schedule(2.seconds, 2.seconds) {
      implicit val timeout = Timeout(5 seconds)
      backend ? GreetNeighbor onSuccess {
        case s => println(s)
      }
    }
  }
}