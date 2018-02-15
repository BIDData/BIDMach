import BIDMach.allreduce._

import scala.concurrent.duration._

val dimNum = 2
val dataSize = 800000
val maxChunkSize = 20000
val roundWorkerPerDimNum = 3
val maxRound = 10000

val threshold = ThresholdConfig(thAllreduce = 1f, thReduce = 1f, thComplete = 1f)
val metaData = MetaDataConfig(dataSize = dataSize, maxChunkSize = maxChunkSize)

val nodeConfig = NodeConfig(dimNum = dimNum, reportStats = true)

val workerConfig = WorkerConfig(
  statsReportingRoundFrequency = 5,
  threshold = threshold,
  metaData = metaData)

val lineMasterConfig = LineMasterConfig(
  roundWorkerPerDimNum = roundWorkerPerDimNum,
  dim = -1,
  maxRound = maxRound,
  workerResolutionTimeout = 5.seconds,
  threshold = threshold,
  metaData = metaData)

AllreduceNode.startUp("0", nodeConfig,lineMasterConfig, workerConfig, assertCorrectness=false, checkpoint = 10)