package BIDMach.allreduce

import akka.actor.ActorRef
import scala.collection.mutable.ArrayBuffer


// worker messages
final case class StartAllreduce(config : RoundConfig)
final case class CompleteAllreduce(srcId : Int, config : RoundConfig)

final case class ScatterBlock(value : Array[Float], srcId : Int, destId : Int, chunkId : Int, config : RoundConfig)
final case class ReduceBlock(value: Array[Float], srcId : Int, destId : Int, chunkId : Int, config : RoundConfig, count: Int)

final case class AllreduceStats(outgoingFloats: Long, incomingFloats: Long)

final case class RoundConfig(lineMasterVersion : Int, round: Int, lineMaster : ActorRef, peers: Map[Int, ActorRef], workerId: Int) {
  def < (other : RoundConfig): Boolean = {
  	return true if (self.lineMasterVersion < other.lineMasterVersion || 
  	  				(self.lineMasterVersion == other.lineMasterVersion && self.round < other.round)) 
  	  else false
  }

  def == (other : RoundConfig): Boolean = {
  	return true if (self.lineMasterVersion == other.lineMasterVersion && self.round == other.round) else false
  }

  def > (other : RoundConfig): Boolean = {
  	return !(self < other || self == other)
  }
}

// GM to LM
/**
  * TODO: Rename this for consistency.
  * This should refer to the node address of [[AllreduceNode]]
  */
final case class StartAllreduceTask(slaveNodesRef: ArrayBuffer[ActorRef], lineMasterVersion : Int)