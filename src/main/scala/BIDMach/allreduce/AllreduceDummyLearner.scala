package BIDMach.allreduce

import java.util.concurrent.Future

import BIDMach.Learner
import BIDMach.datasinks.DataSink
import BIDMach.datasources.DataSource
import BIDMach.mixins.Mixin
import BIDMach.models.Model
import BIDMach.networks.Net
import BIDMach.updaters.Updater

class AllreduceDummyLearner(learner:Learner, dummy_model:AllreduceDummyModel)
  extends Learner(learner.datasource,dummy_model,learner.mixins, learner.updater, learner.datasink ,learner.opts) {

  def this(){
    this(Net.learner("dummy learner")._1, new AllreduceDummyModel())
  }


  override def train: Unit = {
    println("dummy model is training!")
    while(true){
      this.ipass+=1
      this.dummy_model.showSomeWork()
      Thread.sleep(10000)
    }

  }

}
