package BIDMach.allreduce

import akka.actor.ActorRef
import scala.collection.mutable.ArrayBuffer


// worker messages
final case class StartAllreduce(config : RoundConfig)
final case class CompleteAllreduce(srcId : Int, config : RoundConfig)

final case class ScatterBlock(value : Array[Float], srcId : Int, destId : Int, chunkId : Int, config : RoundConfig)
final case class ReduceBlock(value: Array[Float], srcId : Int, destId : Int, chunkId : Int, config : RoundConfig, count: Int)

final case class AllreduceStats(outgoingFloats: Long, incomingFloats: Long)

/**
  * "comparison override to provide a (line master version, round) pair for a smooth transition when nodes are added or removed
  */
final case class RoundConfig(lineMasterVersion : Int, round: Int, lineMaster : ActorRef, peerWorkers: Map[Int, ActorRef], workerId: Int) {
  def < (other : RoundConfig): Boolean = {
  	return if (lineMasterVersion < other.lineMasterVersion || 
  	  		   (lineMasterVersion == other.lineMasterVersion && round < other.round)) {true}
  	else {false}
  }

  def == (other : RoundConfig): Boolean = {
  	return if (lineMasterVersion == other.lineMasterVersion && round == other.round) {true} else {false}
  }

  def > (other : RoundConfig): Boolean = {
  	return !(this < other || this == other)
  }
}

/*
 * Following message used by Line Master
 */
final case class StartAllreduceTask(peerNodes: ArrayBuffer[ActorRef], lineMasterVersion : Int)
final case class StopAllreduceTask(lineMasterVersion : Int)

/*
 * For grid master in case we want to kill the node
 */
final case class StopAllreduceNode()