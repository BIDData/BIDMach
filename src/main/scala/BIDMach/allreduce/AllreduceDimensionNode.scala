package BIDMach.allreduce

import BIDMach.allreduce.AllreduceNode.{DataSink, DataSource}
import akka.actor.{Actor, ActorRef, Props}

/**
  * Generate a set of independent round workers and a line master which may or may not be active depending on the selection.
  *
  * @param roundSources data sources, one for each round worker
  * @param roundSinks   data sinks, one for each round worker
  */
class AllreduceDimensionNode(
                              dimensionNodeConfig: DimensionNodeConfig,
                              lineMasterConfig: LineMasterConfig,
                              workerConfig: WorkerConfig,
                              roundSources: List[DataSource],
                              roundSinks: List[DataSink]
                            ) extends Actor with akka.actor.ActorLogging {

  val assignedDimension = dimensionNodeConfig.dim

  var roundWorkers: Array[ActorRef] = Array.empty
  var lineMaster: Option[ActorRef] = None

  generateRoundWorkers()
  generateLineMaster()

  override def receive: Receive = {
    case _ =>
      log.error(s"\n----DimensionNode!dim=${assignedDimension}: I AM NOT SUPPOSED TO RECEIVE MSGs")
  }

  protected def workerClassProvider(): Class[_] = {
    classOf[AllreduceWorker]
  }

  def generateRoundWorkers(): Unit = {

    if (roundSources.length != lineMasterConfig.roundWorkerPerDimNum || roundSources.length != roundSinks.length) {
      throw new IllegalArgumentException(s"Sources and sinks sizes should correspond to the number of round workers, " +
        s"given ${lineMasterConfig.roundWorkerPerDimNum}, but source size is [${roundSources.length}], and sink [${roundSinks.length}]")
    }
    roundWorkers = {
      val arr = new Array[ActorRef](lineMasterConfig.roundWorkerPerDimNum)
      for (roundNth <- 0 until lineMasterConfig.roundWorkerPerDimNum) {
        val worker = context.actorOf(Props(
          workerClassProvider(),
          workerConfig,
          roundSources(roundNth),
          roundSinks(roundNth)),
          name = s"Worker-round=${roundNth}"
        )
        log.info(s"\n----DimensionNode!dim=${assignedDimension}: Worker for round:$roundNth created with ${worker}")
        arr(roundNth) = worker
      }
      arr
    }
  }

  def generateLineMaster(): Unit = {

    val configWithAssignedDimension = lineMasterConfig.copy(dim = assignedDimension)

    lineMaster = Some(context.actorOf(Props(
      classOf[AllreduceLineMaster],
      configWithAssignedDimension),
      name = "LineMaster"))

    log.info(s"\n----DimensionNode!dim=${assignedDimension}: LineMaster is created with ${lineMaster}")
  }
}