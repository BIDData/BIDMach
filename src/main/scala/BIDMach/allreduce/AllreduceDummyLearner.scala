package BIDMach.allreduce

import BIDMach.Learner
import BIDMach.networks.Net

/**
  * A dummy learner for ease of test. Can be opt or refactored out if necessary
  * @param learner
  * @param dummy_model
  */
class AllreduceDummyLearner(learner:Learner, dummy_model:AllreduceDummyModel)
  extends Learner(learner.datasource,dummy_model,learner.mixins, learner.updater, learner.datasink ,learner.opts) {

  def this(){
    this(Net.learner("dummy learner")._1, new AllreduceDummyModel())
  }


  override def train: Unit = {
    println("dummy model is training!")
    while(true){
      this.ipass+=1
      myLogger.info("pass=%2d" format ipass)
      this.dummy_model.showSomeWork()
    }

  }

}
