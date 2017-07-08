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

/**
 * Layer compute Actor. Attach one of these to each layer. 
 * 
 * On creation, this actor finds the set of Actors attached to this layer's inputs. 
 * For this to work, actors should be created in the layer's DAG order. 
 * 
 * Send "GetOutputs" messages to all layer actors to find node successors. These messages can go in any order, and only 
 * need to be sent once. 
 * 
 * Send "PrepForward" messages to all layer actors at the start of each forward cycle. 
 * These messages can be sent in any order. 
 * 
 * Send "DoForward" messages to layer actors to start forward computation. Order can be arbitrary, but its generally 
 * most efficient to send these in DAG order. These need only be sent to source (indegree zero) nodes. 
 * 
 * Send "PrepBackward" messages to all layer actors at the start of each backward cycle. 
 * These messages can be sent in any order. 
 * 
 * Send "DoBackward" messages to layer actors to start backward computation. Order can be arbitrary, but latency is
 * minimized when message are sent in reverse DAG order. These need only be sent to sink (outdegree zero) nodes. 
 * 
 * In all cases, "ImDone" messages are sent to the parent actor when each node has completed its task. 
 * 
 */

class LayerActor(val layer:Layer) extends Actor {
  import LayerActor._
  layer.myActor = self;
  val inputActors = Set(layer.inputs.map(_.layer.myActor):_*);
  val ninputs = inputActors.size;
  var outputActors = Set.empty[ActorRef];
  var todoInputs = Set.empty[ActorRef];
  var todoOutputs = Set.empty[ActorRef];
  def noutputs = outputActors.size;
  
  def receive = {

    case GetOutputs => {
      if (ninputs == 0) {
      	context.parent ! ImDone;       
      } else {
      	todoInputs = inputActors.clone;
      	for (a <- inputActors) {
      		a ! IsOutput;
      	}
      }
    }
    
    case IsOutput => {
      outputActors += sender();
      sender() ! AckIsOutput;
    } 
    
    case AckIsOutput => {
      todoInputs -= sender();
      if (todoInputs.isEmpty) {
      	context.parent ! ImDone;      
      }
    }
    
    case PrepForward => {
    	if (ninputs > 0) {
    		todoInputs = inputActors.clone;
    	}
    	context.parent ! ImDone;
    }
    
    case DoForward => {
      if (ninputs == 0) {
        layer.forward;
        for (a <- outputActors) {
          a ! ReadyForward;
        }
        context.parent ! ImDone;
      }
    }
    
    case ReadyForward => {
      todoInputs -= sender();
      if (todoInputs.isEmpty) {
        layer.forward;
        for (a <- outputActors) {
          a ! ReadyForward;
        }
        context.parent ! ImDone;
      }
    }
    
    case PrepBackward => {
    	if (noutputs > 0) {
    		todoOutputs = outputActors.clone;
    	}
    	context.parent ! ImDone;
    }
    
    case DoBackward => {
      if (noutputs == 0) {
        layer.backward;
        for (a <- inputActors) {
          a ! ReadyBackward;
        }
        context.parent ! ImDone;
      } 
    }
    
    case ReadyBackward => {
      todoOutputs -= sender();
      if (todoOutputs.isEmpty) {
        layer.backward;
        for (a <- inputActors) {
          a ! ReadyBackward;
        }
        context.parent ! ImDone;
      }   
    }
    

    case _ => {}
  }
}

object LayerActor {
  case object GetOutputs;
  case object IsOutput;
  case object AckIsOutput
  case object PrepForward;
  case object DoForward;
  case object ReadyForward;
  case object PrepBackward;
  case object DoBackward;
  case object ReadyBackward;;
  case object ImDone;
  def props(layer: Layer): Props = Props(new LayerActor(layer));
}