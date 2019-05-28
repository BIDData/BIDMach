package BIDMach.networks.layers


import BIDMat.{Mat,SBMat,CMat,DMat,FMat,IMat,LMat,HMat,GMat,GDMat,GIMat,GLMat,GSMat,GSDMat,SMat,SDMat,TMat,FFilter,Filter,GFilter}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import BIDMach.datasources._
import BIDMach.updaters._
import BIDMach.mixins._
import BIDMach.models._
import BIDMach._
import edu.berkeley.bid.CPUMACH
import edu.berkeley.bid.CUMACH
import jcuda._
import jcuda.runtime._
import jcuda.runtime.JCuda._
import jcuda.jcudnn._
import jcuda.jcudnn.JCudnn._
import scala.util.hashing.MurmurHash3
import java.util.HashMap
import BIDMach.networks._
import java.util.Arrays

@SerialVersionUID(100L)
class ElasticLayer(override val net:Net, override val opts:ElasticNodeOpts = new ElasticNode ) extends Layer(net, opts) {

  var reducedmats:Array[Mat] = null;
  var dataToAllReduce:Array[Float] = null;
  var dataFromAllReduce:Array[Float] = null;
  var onGPU = false;
  var block = 512;
  
	def initReducedMats {
	  reducedmats = net.modelmats.map(_.copy);
	  val totalsize0 = net.modelmats.map(_.length).reduce(_+_);
	  val totalsize = block * (1 + (totalsize0 - 1)/block);
	  val packedSrc = if (onGPU) GMat(block, totalsize/block) else FMat(block, totalsize/block);
	  var istart = 0;
	  for (i <- 0 until reducedmats.length) {
	  	val len = net.modelmats(i).length;
	    (net.modelmats(i), packedSrc) match {
	      case (fromdata:GMat, todata:GMat) => {	    
	      	cudaMemcpy(todata.pdata.withByteOffset(istart * 4), fromdata.pdata, len * 4, cudaMemcpyKind.cudaMemcpyDeviceToDevice);
	      }
	      case (fromdata:GMat, todata:FMat) => {	    
	      	cudaMemcpy(Pointer.to(todata.data).withByteOffset(istart * 4), fromdata.pdata, len * 4, cudaMemcpyKind.cudaMemcpyDeviceToHost);
	      }
	      case (fromdata:FMat, todata:FMat) => {	    
	      	System.arraycopy(fromdata.data, 0, todata.data, istart, len);
	      }
          case _ => throw new RuntimeException("ElasticLayer initReduceMats matrix type not matched");
	    }
	    istart += len;
	  } 
	}
	
	
	override def forward = {
    val start = toc; 
    if (reducedmats.asInstanceOf[AnyRef] == null) initReducedMats;
    
    /* start an allreduce in here */
    
    forwardtime += toc - start;
	}
	
	override def backward = {
    val start = toc; 
    
    /* allreduce hopefully done */
    
    val modelmats = net.modelmats;
    
    for (i <- 0 until modelmats.length) {
      var diff:Mat = null;
      reducedmats(i).synchronized {
      	diff = reducedmats(i) - modelmats(i);
      }
      diff ~ diff *@ opts.elastic_weight;
      modelmats(i) ~ modelmats(i) + diff;
    }
    
    backwardtime += toc - start;
	}
	
	
	
  override def toString = {
    "Elastic@" + Integer.toHexString(hashCode() % 0x10000)
  }
  
}

trait ElasticNodeOpts extends ModelNodeOpts {
  var elastic_weight = 0f

  def copyOpts(opts:ElasticNodeOpts):ElasticNodeOpts = {
  	super.copyOpts(opts);
  	opts.elastic_weight = elastic_weight;
  	opts;
  }

}

@SerialVersionUID(100L)
class ElasticNode extends Node with ElasticNodeOpts {

  def copyTo(opts:ElasticNode):ElasticNode = {
    this.asInstanceOf[Node].copyTo(opts);
    copyOpts(opts);
    opts
  }

  override def clone:ElasticNode = {
    copyTo(new ElasticNode ).asInstanceOf[ElasticNode]
  }
  
  override def create(net:Net):ElasticLayer = {
    ElasticLayer(net, this)
  }

  override def toString = {
    "Elastic@" + Integer.toHexString(hashCode() % 0x10000)
  }


}

@SerialVersionUID(100L)
object ElasticLayer {
  
  def apply(net:Net) = new ElasticLayer(net, new ElasticNode);
  
  def apply(net:Net, opts:ElasticNodeOpts) = new ElasticLayer(net, opts);

}
