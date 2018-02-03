package BIDMach.allreduce

import scala.concurrent.duration._

case class MasterConfig(
                         nodeNum: Int,
                         discoveryTimeout: FiniteDuration)

case class ThresholdConfig(
                            thAllreduce: Float,
                            thReduce: Float,
                            thComplete: Float)

case class MetaDataConfig(
                           dataSize: Int,
                           maxChunkSize: Int)

case class LineMasterConfig(
                             workerPerNodeNum: Int,
                             dim: Int,
                             maxRound: Int,
                             discoveryTimeout: FiniteDuration,
                             threshold: ThresholdConfig,
                             metaData: MetaDataConfig)

case class NodeConfig(dimNum: Int)

case class WorkerConfig(
                         discoveryTimeout: FiniteDuration,
                         threshold: ThresholdConfig,
                         metaData: MetaDataConfig)

case class DimensionNodeConfig(
                                dim: Int)