package BIDMach.allreduce

import BIDMat.{Mat,SBMat,CMat,DMat,FMat,IMat,HMat,GDMat,GLMat,GMat,GIMat,GSDMat,GSMat,LMat,SMat,SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import edu.berkeley.bid.comm._
import scala.collection.parallel._
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.net.ServerSocket;
import java.net.Socket;
import java.net.SocketException;
import java.io.DataInputStream;
import java.io.IOException;

class Worker(val opts:Worker.Opts) extends Serializable {
  /*
   * Constructor arguments: 
   *   M: number of machines
   *   F: number of features
   *   nnz: number of non-zero features (expected) per node
   *   
   * AllReduce arguments:
   *   stride: number of rows in the matrices to reduce
   *   
   */
  var M:Int = 0;	
	var gmods:IMat = null;
	var gridmachines:IMat = null;        
	var groups:Groups = null;

	var machExecutor:ExecutorService = null;
	var imach = 0;
	var replicate:Int = 1;
	var machine:Machine = null;
	var bufsize = 10*1000000;

  
  def config(imach0:Int, gmods0:IMat, gridmachines0:IMat, machineIPs:Array[String]) = {    
    imach = imach0;
    gmods = gmods0;
    gridmachines = gridmachines0;
    M = gridmachines.length;
    groups = new Groups(M, gmods.data, gridmachines.data, 0);
    
    var totvals = 0L;

    machine = new Machine(null, groups, imach, M, opts.useLong, bufsize, false, opts.trace, replicate, machineIPs);
    machExecutor = machine.executor;
    machine.configTimeout = opts.configTimeout;
    machine.reduceTimeout = opts.reduceTimeout;
    machine.sendTimeout = opts.sendTimeout;
    machine.recvTimeout = opts.recvTimeout;
  }
  
  def allReduce(round:Int, stride:Int) = {

/*  	val result = if (fuseConfigReduce) {
  		if (useLong) {
  			machine.configReduce(nodeIndsLong(imach).data, nodeIndsLong(imach).data, nodeData(imach).data, stride, round);
  		} else {
  			machine.configReduce(nodeInds(imach).data, nodeInds(imach).data, nodeData(imach).data, stride, round);
  		}
  	} else {
  		if (useLong) {
  			machine.config(nodeIndsLong(imach).data, nodeIndsLong(imach).data, round);
  		} else {
  			machine.config(nodeInds(imach).data, nodeInds(imach).data, round); 
  		}
  		machine.reduce(nodeData(imach).data, stride, round);
  	}*/
  }
  
  def shutdown = {
    machExecutor.shutdownNow();
    val tt= toc;
  }
  
  def handleCMD(cmd:Command) = {
  }
  
  def log(ss:String) = {
    machine.log(ss);
  }
}

object Worker {
	trait Opts extends BIDMat.Opts{
		var configTimeout = 3000;
		var reduceTimeout = 3000;
		var sendTimeout = 1000;
		var recvTimeout = 1000;
		var cmdTimeout = 1000;
		var fuseConfigReduce = false;
		var trace = 0;
		var useLong = false;
  }
	
	class Options extends Opts {}
}