import BIDMach.allreduce._

import scala.concurrent.duration._

val workerPerNodeNum = 3
val dataSize = 100

val maxChunkSize = 4

val threshold = ThresholdConfig(thAllreduce = 1f, thReduce = 1f, thComplete = 0.8f)
val metaData = MetaDataConfig(dataSize = dataSize, maxChunkSize = maxChunkSize)

val workerConfig = WorkerConfig(workerPerNodeNum = workerPerNodeNum,
  discoveryTimeout = 5.seconds,
  threshold = threshold,
  metaData= metaData)

AllreduceNode.startUp("0", workerConfig)