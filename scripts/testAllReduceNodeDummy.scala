import BIDMach.allreduce.AllreduceNode.getBasicConfigs
import BIDMach.allreduce.binder.AssertCorrectnessBinder
import BIDMach.allreduce.{AllreduceDummyLearner, AllreduceNode}

val learner = new AllreduceDummyLearner()
learner.ipass = 20

val dataSize = 5000000
val maxChunkSize = 7779

val basicConfig = getBasicConfigs()
val modifiedConfig = basicConfig.copy(workerConfig =
  basicConfig.workerConfig.copy(
    metaData = basicConfig.workerConfig.metaData.copy(dataSize = dataSize, maxChunkSize = maxChunkSize)
  )
)


val binder = new AssertCorrectnessBinder(dataSize, 10)
AllreduceNode.startNodeAfterIter(learner = learner, iter = 0, nodeConfig = modifiedConfig, binder = binder)