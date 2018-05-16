package BIDMach.allreduce

import BIDMach.allreduce.AllreduceType.AllreduceType

import scala.concurrent.duration._

/**
  * @param nodeNum how many node is in the network
  * @param nodeResolutionTimeout resolution for node seek process
  */
case class GridMasterConfig(
                             nodeNum: Int,
                             nodeResolutionTimeout: FiniteDuration) {

}

/**
  * Threshold config at various stages of all-reduce
  *
  * @param thAllreduce th percentage of the peers sending completion at which line master will start a new all reduce round
  * @param thReduce    th percentage of the scatter messages received per chunk at which the worker will reduce results
  * @param thComplete  th percentage of the reduced messages received across the data size at which worker will signal completion to the line master
  */
case class ThresholdConfig(
                            thAllreduce: Float,
                            thReduce: Float,
                            thComplete: Float)

/**
  *  Hyperparameter for the whole process, sharing across machines
  *
  * @param dataSize the size of the array to have all reduce operation on
  * @param maxChunkSize the chunk size to be sent as message during communication. Might be smaller if not split evenly.
  */
case class MetaDataConfig(
                           dataSize: Int = -1,
                           maxChunkSize: Int)

/**
  * @param roundWorkerPerDimNum for one dimension, how many round worker we should have
  * @param dim ususally 2, extendable
  * @param maxRound upper limit of the round upon which the all reduce process would stop
  * @param workerResolutionTimeout timeout used for worker seeking process
  * @param threshold
  */
case class LineMasterConfig(
                             roundWorkerPerDimNum: Int,
                             dim: Int,
                             maxRound: Int,
                             workerResolutionTimeout: FiniteDuration,
                             threshold: ThresholdConfig)

/**
  * Parameters self-explanatory
  * @param statsReportingRoundFrequency
  * @param threshold
  * @param metaData
  * @param reducer
  */
case class WorkerConfig(
                         statsReportingRoundFrequency: Int = 10,
                         threshold: ThresholdConfig,
                         metaData: MetaDataConfig,
                         reducer: AllreduceType = AllreduceType.Average)

/**
  * @param dim which dimension the node belongs to
  */
case class DimensionNodeConfig(dim: Int)

case class NodeConfig(
                       workerConfig: WorkerConfig,
                       lineMasterConfig: LineMasterConfig,
                       dimNum: Int, reportStats: Boolean, elasticRate: Float
                     )
