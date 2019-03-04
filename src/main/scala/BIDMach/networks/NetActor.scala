package BIDMach.networks

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
 * Net compute Actor. Attach to the network to support async and GPU-streamed layer updates.  
 * 
 * Call setup() once after creation to find outputs for each layer, and source and sink layers.
 * 
 * Supports forward and backward calls which block until the calculations are complete. 
 * 
 */

@SerialVersionUID(100L)
class NetActor(val net:Net) extends Actor {
  import LayerActor._
  
  val actors = net.layers.map((x)=>context.actorOf(LayerActor.props(x)));
  val actorSet =  Set(actors:_*);
  @volatile var todo = Set.empty[ActorRef];
  var sources:Array[ActorRef] = null;
  var sinks:Array[ActorRef] = null;
  
  def setup() = {
    todo = actorSet.clone;
    actors.foreach(_ ! GetOutputs);
    waitComplete;
    sources = actors.filter(_.asInstanceOf[LayerActor].ninputs == 0);
    sinks = actors.filter(_.asInstanceOf[LayerActor].noutputs == 0).reverse;
  }
    
  def forward() = {
    prepForward();
    doForward();
  }
  
  def backward() = {
    prepBackward();
    doBackward();
  }
  
  def receive = {  
    case ImDone => {
      todo.synchronized {
      	todo -= sender();
      }
    }
    case _ => {}
  }
  
  def waitComplete = {
  	var todoIsEmpty = false;
    while (! todoIsEmpty) {
    	todo.synchronized {
      	todoIsEmpty = todo.isEmpty ;
    	}
      Thread.sleep(0,1000);
    } 
  }
  
  def prepForward() = {
  	todo = actorSet.clone;
    actors.foreach(_ ! PrepForward);
    waitComplete;
  }
  
  def doForward() = {
    todo = actorSet.clone;
    sources.foreach(_ ! DoForward);
    waitComplete;
  }
  
  def prepBackward() = {
    todo = actorSet.clone;
    actors.foreach(_ ! PrepBackward);
    waitComplete;
  }
  
  def doBackward() = {
    todo = actorSet.clone;
    sinks.foreach(_ ! DoBackward);
    waitComplete;
  }
}

@SerialVersionUID(100L)
object NetActor {
  def props(net: Net): Props = Props(new NetActor(net));
}
