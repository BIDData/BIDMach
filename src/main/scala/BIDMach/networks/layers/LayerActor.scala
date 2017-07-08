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
import akka.actor.{Actor,Props,ActorSystem,ActorRef};

class LayerActor(val layer:Layer) extends Actor {
  import LayerActor._
  layer.myActor = self;
  val inputActors = layer.inputs.map(_.layer.myActor).distinct;
  val ninputs = inputActors.size;
  var outputActors = Set[ActorRef]();
  def noutputs = outputActors.size;
  var recvdInputs = 0;
  var recvdOutputs = 0;
  
  def receive = {
    case DoForward => {
      recvdInputs = 0;
      if (ninputs == 0) {
        layer.forward;
        for (a <- outputActors) {
          a ! ReadyForward;
        }
      }
      sender() ! ImDone;
    }
    case ReadyForward => {
      recvdInputs += 1;
      if (recvdInputs == ninputs) {
        layer.forward;
        for (a <- outputActors) {
          a ! ReadyForward;
        }
      }
    }
    case DoBackward => {
      recvdOutputs = 0;
      if (noutputs == 0) {
        layer.backward;
        for (a <- inputActors) {
          a ! ReadyBackward;
        }
      }  
      sender() ! ImDone;
    }
    case ReadyBackward => {
      recvdOutputs += 1;
      if (recvdOutputs == noutputs) {
        layer.backward;
        for (a <- inputActors) {
          a ! ReadyBackward;
        }
      }   
    }
    case GetOutputs => {
      for (a <- inputActors) {
        a ! IsOutput;
      }
      sender() ! ImDone;
    }
    case IsOutput => {
      outputActors.add(sender());
    }    
    case _ => {}
  }
}

object LayerActor {
  case object DoForward;
  case object DoBackward;
  case object ReadyForward;
  case object ReadyBackward;
  case object GetOutputs;
  case object IsOutput;
  case object ImDone;
  def props(layer: Layer): Props = Props(new LayerActor(layer));
}