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

class Worker() extends Serializable {
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
	var configDir = "/code/BIDMat/data/bestAllreduce/";
	
	var gmods:IMat = null;
	var gridmachines:IMat = null;        
	var groups:Groups = null;

	var machExecutor:ExecutorService = null;
	var configTimeout = 3000;
	var reduceTimeout = 3000;
  var sendTimeout = 1000;
	var recvTimeout = 1000;
	var cmdTimeout = 1000;
	var fuseConfigReduce = false;
	var trace = 0;
	var useLong = false;
	var replicate:Int = 1;
	var nreps = 1;
	var machine:Machine = null;
	var bufsize = 10*1000000;
	var imach = 0;
	
	def setNumMachines(M0:Int) {
	  M = M0;
	}
	
	def readConfig(configDir:String) {
		val clengths = loadIMat(configDir + "dims.imat.lz4");
		val allgmods = loadIMat(configDir + "gmods.imat.lz4");
		val allmachinecodes = loadIMat(configDir + "machines.imat.lz4");
		gmods = allgmods(0->clengths(M-1), M-1);
		gridmachines = allmachinecodes(0->M, M-1);
	}
  
  def config(imach0:Int, gmods0:IMat, gridmachines0:IMat, machineIPs:Array[String]) = {    
    gmods = gmods0;
    imach = imach0;
    gridmachines = gridmachines0;
    groups = new Groups(M, gmods.data, gridmachines.data, 1000);
    
    var totvals = 0L;

    machine = new Machine(null, groups, imach, M, useLong, bufsize, false, trace, replicate, machineIPs);
    machExecutor = machine.executor;
    machine.configTimeout = configTimeout;
    machine.reduceTimeout = reduceTimeout;
    machine.sendTimeout = sendTimeout;
    machine.recvTimeout = recvTimeout;
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
