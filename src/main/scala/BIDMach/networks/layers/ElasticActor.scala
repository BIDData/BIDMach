package BIDMach.networks.layers

import BIDMat.{Mat,SBMat,CMat,CSMat,DMat,FMat,IMat,LMat,HMat,GFilter,GMat,GDMat,GIMat,GLMat,GSMat,GSDMat,JSON,ND,SMat,SDMat,TMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import BIDMach.datasources._
import BIDMach.datasinks._
import BIDMach.updaters._
import BIDMach.mixins._
import BIDMach.models._
import BIDMach._
import BIDMach.networks.layers._
import jcuda.jcudnn._
import jcuda.jcudnn.JCudnn._
import scala.collection.mutable.Set;
import java.util.HashMap;
import akka.actor.{Actor,Props,ActorSystem,ActorRef,ActorSelection};

/**
 * Elastic Layer compute Actor. Attach one of these to an Elastic Layer. 
 * 
 *
 * 
 */

class ElasticActor(val layer:ElasticLayer) extends Actor {
  import ElasticActor._
  layer.myActor = self;
  var friends:Array[ActorRef] = null;
  var friendSelection:Array[ActorSelection] = null;
  var friendAddresses:Array[String] = null;
  
  def receive = {


    case _ => {}
  }
}

object ElasticActor {

  def props(layer: ElasticLayer): Props = Props(new ElasticActor(layer));
}
