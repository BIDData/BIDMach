package BIDMach.allreduce

import akka.actor.ActorRef


// worker messages
final case class InitWorkers(
	workers: Map[Int, ActorRef],
	workerNum: Int,
	master : ActorRef,
	destId : Int, 
	thReduce : Float, 
	thComplete : Float,
	maxLag : Int,
	dataSize: Int,
	maxChunkSize: Int
)
final case class StartAllreduce(round : Int)
final case class PrepareAllreduce(round: Int, nodeAddresses: Map[Int, ActorRef], nodeId: Int)
final case class ConfirmPreparation(round: Int)
final case class CompleteAllreduce(srcId : Int, round : Int)


final case class ScatterBlock(value : Array[Float], srcId : Int, destId : Int, chunkId : Int, round : Int)
final case class ReduceBlock(value: Array[Float], srcId : Int, destId : Int, chunkId : Int, round : Int, count: Int)
