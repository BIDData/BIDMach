package BIDMach.allreduce

import BIDMach.allreduce.AllreduceType.AllreduceType

import scala.concurrent.duration._


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

case class MetaDataConfig(
                           dataSize: Int = -1,
                           maxChunkSize: Int)

case class LineMasterConfig(
                             roundWorkerPerDimNum: Int,
                             dim: Int,
                             maxRound: Int,
                             workerResolutionTimeout: FiniteDuration,
                             threshold: ThresholdConfig)

case class WorkerConfig(
                         statsReportingRoundFrequency: Int = 10,
                         threshold: ThresholdConfig,
                         metaData: MetaDataConfig,
                         reducer: AllreduceType = AllreduceType.Average)

case class DimensionNodeConfig(dim: Int)

case class NodeConfig(
                       workerConfig: WorkerConfig,
                       lineMasterConfig: LineMasterConfig,
                       dimNum: Int, reportStats: Boolean, elasticRate: Double
                     )
