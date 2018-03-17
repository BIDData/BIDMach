import BIDMach.allreduce.binder.AssertCorrectnessBinder
import BIDMach.allreduce.{AllreduceDummyLearner, AllreduceNode, AllreduceTrainer}

val learner = new AllreduceDummyLearner()
learner.launchTrain

AllreduceNode.start(learner)

