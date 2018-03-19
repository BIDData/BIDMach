package BIDMach.allreduce

import BIDMach.models.Model
import BIDMat.{FMat, Mat}

class AllreduceDummyModel(val _modelmat: Array[Mat]) extends Model {
  def this(){
    this(Array[Mat](FMat.ones(30,100),FMat.ones(100,30)))
  }


  override def modelmats:Array[Mat] = {
    _modelmat
  }
  override def init()={}
  override def dobatch(mats:Array[Mat], ipass:Int, here:Long)={}
  override def evalbatch(mats: Array[Mat], ipass: Int, here:Long):FMat = {
    FMat.zeros(0,0)
  }
  def showSomeWork(){
    println("I'm learning something")
    Thread.sleep(1000)
  }

}
