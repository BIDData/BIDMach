import BIDMach.allreduce._

import scala.concurrent.duration._

val workerNum = 3
val dataSize = 100

val maxChunkSize = 4

val threshold = ThresholdConfig(thAllreduce = 1f, thReduce = 1f, thComplete = 0.8f)
val metaData = MetaDataConfig(dataSize = dataSize, maxChunkSize = maxChunkSize)
//
val workerConfig = WorkerConfig(workerNum = workerNum,
  discoveryTimeout = 5.seconds,
  threshold = threshold,
  metaData= metaData)

AllreduceNode.startUp("0", workerConfig)