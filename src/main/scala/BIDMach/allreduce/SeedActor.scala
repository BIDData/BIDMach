package BIDMach.allreduce

import akka.actor.{Actor,ActorRef,Props,Address,ActorSystem}
import akka.cluster.{Cluster,MemberStatus}	
import akka.cluster.ClusterEvent._
import com.typesafe.config.ConfigFactory
import akka.event.Logging
import akka.actor.ActorLogging
import scala.collection.mutable.ListBuffer
import SeedActor._


class SeedActor extends Actor with ActorLogging {

  val cluster = Cluster(context.system)

  var nodes = Set.empty[Address];

  // subscribe to cluster changes, re-subscribe when restart 
  override def preStart(): Unit = {
    cluster.subscribe(self, classOf[MemberEvent], classOf[UnreachableMember])
    }

  override def postStop(): Unit = cluster.unsubscribe(self);

  def receive = {
  case state: CurrentClusterState => {
	  nodes = state.members.collect {
	  case m if m.status == MemberStatus.Up => m.address
	  }
	  log.info("Member init: {} nodes", nodes.size);
      }
  case MemberUp(member) => {
      nodes += member.address;
      log.info("Member is Up: {}", member.address);
  }
  case UnreachableMember(member) => {
      nodes -= member.address;
      log.info("Member detected as unreachable: {}", member);
  }
  case MemberRemoved(member, previousStatus) => {
      nodes -= member.address;
      log.info("Member is Removed: {} after {}", member.address, previousStatus)
  }
  case _: MemberEvent => {}// ignore
  case x:GetNodes => {
      x.nodes = nodes;
  }
  }
}

object SeedActor {
val actorSystems = new ListBuffer[ActorSystem];

val actors = new ListBuffer[ActorRef];

case class GetNodes() {
  var nodes:Set[Address] = null;

  def query(a:ActorRef) = {
      a ! this;
      nodes;
  }	
}

def startup(ports: Seq[String]) = {
    ports map { port =>
		// Override the configuration of the port
		val config = ConfigFactory.parseString("akka.remote.netty.tcp.port=" + port).
		withFallback(ConfigFactory.load())
		
		// Create an Akka system
		val system = ActorSystem("ClusterSystem", config)
		actorSystems += system
		// Create an actor that handles cluster domain events
		val actor = system.actorOf(Props[SeedActor], name = "clusterListener")
		actors += actor
		actor
    }
}

def shutdown = {
   actorSystems.map(_.terminate);
}
}