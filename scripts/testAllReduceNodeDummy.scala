import BIDMach.allreduce.AllreduceNode.getBasicConfigs
import BIDMach.allreduce.binder.{AssertCorrectnessBinder, NoOpBinder}
import BIDMach.allreduce.{AllreduceDummyLearner, AllreduceNode}

val learner = new AllreduceDummyLearner()
learner.ipass = 20

val dataSize = 60000000
val maxChunkSize = 20000

val basicConfig = getBasicConfigs()
val modifiedConfig = basicConfig.copy(workerConfig =
  basicConfig.workerConfig.copy(
    metaData = basicConfig.workerConfig.metaData.copy(dataSize = dataSize, maxChunkSize = maxChunkSize),
    threshold =  basicConfig.workerConfig.threshold.copy(thComplete = 1.0f)
  )
)


val binder = new NoOpBinder(dataSize, 10)
AllreduceNode.startNodeAfterIter(learner = learner, iter = 0, nodeConfig = modifiedConfig, binder = binder)