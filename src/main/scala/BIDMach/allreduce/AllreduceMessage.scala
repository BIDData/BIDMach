package BIDMach.allreduce

import akka.actor.ActorRef
import scala.collection.mutable.ArrayBuffer


// worker messages
final case class StartAllreduce(round : Int)
final case class PrepareAllreduce(round: Int, workerAddresses: Map[Int, ActorRef], workerId: Int)
final case class ConfirmPreparation(round: Int)
final case class CompleteAllreduce(srcId : Int, round : Int)

final case class ScatterBlock(value : Array[Float], srcId : Int, destId : Int, chunkId : Int, round : Int)
final case class ReduceBlock(value: Array[Float], srcId : Int, destId : Int, chunkId : Int, round : Int, count: Int)

// GM to LM
final case class SlavesInfo(slaveNodesRef: ArrayBuffer[ActorRef])