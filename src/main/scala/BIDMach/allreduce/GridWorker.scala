package BIDMach.allreduce

import akka.actor.{Actor, ActorRef, ActorSystem, Props, Terminated}
import akka.pattern.ask
import akka.util.Timeout
import com.typesafe.config.ConfigFactory

import scala.concurrent.ExecutionContext.Implicits.global
import scala.concurrent.duration._
import scala.language.postfixOps

class GridWorker extends Actor {

  var myGroups: collection.mutable.Map[GridGroup, collection.mutable.Set[ActorRef]] = collection.mutable.Map.empty[GridGroup, collection.mutable.Set[ActorRef]]

  def receive = {

    // Do work
    case GreetGroups() =>
      for((group_name, group)  <- myGroups){
        for(member <- group) {
          if (member != self) {
            member ! HelloFromGroup(s"Index: ${group_name.index}, Dimension: ${group_name.dim}")
          }
        }
      }

    case HelloFromGroup(msg) =>
      println(s"Receiving greeting from ${sender()} msg: $msg")

    // Manage neighbors
    case GridGroupAddresses(group, addresses) =>
      if(!myGroups.contains(group)){
        myGroups(group) = collection.mutable.Set(addresses.toArray:_*)
        for(address <- addresses){
          context.watch(address)
        }
      } else{
        println(s"Found duplicate group given: $group with $addresses")
      }

    case Terminated(a) =>
      println(s"Neighbor $a is terminated, removing it from all the groups")
      for((group_name, group)  <- myGroups){
        if(group.contains(a)) group -= a
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
    //src/resources/conf/application.conf

    val system = ActorSystem("ClusterSystem", config)
    val worker = system.actorOf(Props[GridWorker], name = "worker")

    system.scheduler.schedule(2.seconds, 2.seconds) {
      implicit val timeout = Timeout(5 seconds)
      worker ? GreetGroups() onSuccess {
        case s => println(s)
      }
    }
  }

  def startUp() = {
    main(Array())
  }


}