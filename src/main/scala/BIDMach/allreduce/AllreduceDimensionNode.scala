package BIDMach.allreduce

import BIDMach.allreduce.AllreduceWorker.{DataSink, DataSource}
import akka.actor.{Actor, ActorRef, ActorSystem, Props}
import com.typesafe.config.ConfigFactory
import scala.collection.mutable.ArrayBuffer



class AllreduceDimensionNode(
					          dimensionNodeConfig: DimensionNodeConfig,
                    lineMasterConfig: LineMasterConfig,
                    workerConfig: WorkerConfig,
                    sources: List[DataSource],
                    sinks: List[DataSink]
                ) extends Actor with akka.actor.ActorLogging {

  var workers: Array[ActorRef] = Array.empty
  var lineMaster: Option[ActorRef] = None
  val dim = dimensionNodeConfig.dim

  generateWorkers()
  generateLineMaster()

  override def receive: Receive = {
    case _ => 
      log.error(s"\n----DimensionNode!dim=${dim}: I AM NOT SUPPOSED TO RECEIVE MSGs")
  }

  def generateWorkers(): Unit = {
    workers = {
      val arr = new Array[ActorRef](lineMasterConfig.roundNum)
      for (i <- 0 until lineMasterConfig.roundNum) {
	      val worker = context.actorOf(Props(
	        classOf[AllreduceWorker],
	        workerConfig,
	        sources(i),
	        sinks(i)),
	        name = s"Worker-id=${i}"
	      )
	      log.info(s"\n----DimensionNode!dim=${dim}: Worker for round:$i created with ${worker}")
	      arr(i) = worker
      }
      arr
    }
  }

  def generateLineMaster(): Unit = {
  	var lineMasterConfig_ = LineMasterConfig(
          roundNum          = lineMasterConfig.roundNum,
          dim               = dim,
          maxRound          = lineMasterConfig.maxRound,
          discoveryTimeout  = lineMasterConfig.discoveryTimeout,
          threshold         = lineMasterConfig.threshold,
          metaData          = lineMasterConfig.metaData)

    lineMaster = Some(context.actorOf(Props(
      classOf[AllreduceLineMaster],
      lineMasterConfig_),
      name = "LineMaster"))

    log.info(s"\n----DimensionNode!dim=${dim}: LineMaster is created with ${lineMaster}")
  }
}