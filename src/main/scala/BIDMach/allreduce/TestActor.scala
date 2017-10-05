package BIDMach.allreduce

import akka.actor.{Actor,ActorRef,Props,Address,ActorSystem,ActorSelection}
import akka.cluster.{Cluster,MemberStatus}	
import akka.cluster.ClusterEvent._
import BIDMat.{BMat,Mat,SBMat,CMat,DMat,FMat,FFilter,IMat,HMat,GDMat,GFilter,GLMat,GMat,GIMat,GSDMat,GSMat,LMat,SMat,SDMat,TMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import com.typesafe.config.ConfigFactory
import akka.event.Logging
import akka.actor.ActorLogging
import TestActor._


class TestActor extends Actor with ActorLogging {
    var v = 0;
    var t1 = 0f;
    var n0 = 0;
    
    def receive = {
    case x:SendTo => {
	    val sel = context.actorSelection(x.dest);
	    sel ! new SendVal(x.v);
	}
    case x:SendVal => {
	    val m = new RecvVal(x.v * 2);
	    println("%s got a sendval msg %d from %s" format (self, x.v, sender().toString));
	    sender ! m;
	}
    case x:RecvVal => {
    	    println("%s got a recval msg %d from %s" format (self, x.v, sender().toString));
	    v = x.v;
	}
    case x:GetVal => {
	    x.v = v;
	}
    case x:DataPacket => {
	    if (x.n > 0) {
		val y = new DataPacket(x.d, x.n-1);
		sender ! y;
	    } else {
		t1 = toc;
		val bytes = x.d.length * n0 * 4.0;
		println("%2.1f Mbytes in %2.1f seconds at %4.3f MBytes/sec" format (bytes/1.0e6, t1, bytes/1.0e6/t1));
	    }
	}
    case x:SendData => {
	    val sel = context.actorSelection(x.dest);
	    n0 = x.n;
	    tic;
	    sel ! new DataPacket(x.d, x.n-1);
	}
    case _ => {}
    }
}


object TestActor {

case class SendVal(val v:Int) {}

case class RecvVal(val v:Int) {}

case class SendTo(val dest:String, val v:Int) {}

case class GetVal() {var v = 0;}

case class DataPacket(val d:FMat, val n:Int) {}

case class SendData(val dest:String, val d:FMat, val n:Int) {}

def startup(ports: Seq[String]) = {
    ports map { port =>
		// Override the configuration of the port
		val config = ConfigFactory.parseString("akka.remote.netty.tcp.port=" + port).
		withFallback(ConfigFactory.load())
		
		// Create an Akka system
		val system = ActorSystem("ClusterSystem", config)
		// Create an actor that handles cluster domain events
		system.actorOf(Props[TestActor], name = "testActor")
    }
}

}
