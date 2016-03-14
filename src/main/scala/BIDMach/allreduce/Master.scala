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

class Master() extends Serializable {
  
  var M = 0;
  var sendTimeout = 1000;
	var recvTimeout = 1000;
	var trace = 0;
	var gmods:IMat = null;
	var gridmachines:IMat = null;
	var workerIPs:IMat = null;
  
  def configNodes(gmods0:IMat, gridmachines0:IMat, machineIPs:IMat) {
    gmods = gmods0;
    gridmachines = gridmachines0;
    workerIPs = machineIPs; 
    M = machineIPs.length;
    val clen = 4 + gmods.length + gridmachines.length + workerIPs.length;
    val cmd = new ConfigCommand(clen, new Array[Byte](4*clen));
    cmd.gmods = gmods;
    cmd.gridmachines = gridmachines;
    cmd.workerIPs = workerIPs;
    for (imach <- 0 until M) {
      cmd.imach = imach;
      cmd.encode;
      send(cmd, workerIPs(imach));      
    }
  }
  
  def permuteNodes(seed:Long) {
    
  }
  
  def startUpdates() {
    
  }
  
  def stopUpdates() {
    
  }
  
  def log(msg:String) {
		print(msg);	
	}
  
  def send(cmd:Command, address:Int) = {
    
  }
}

object AllReduceMaster {
	trait Opts extends BIDMat.Opts{
		var limit = 0;
		var limitFctn:(Int,Int)=>Int = null;
  }
	
	class Options extends Opts {} 
	
	def powerLimit(round:Int, limit:Int, power:Float):Int = {
	  if (round < 2) {
	    limit
	  } else {
	    var rnd = round;
	    var nzeros = 0;
	    while ((rnd & 1) == 0) {
	      rnd = (rnd >> 1);	  
	      nzeros += 1;
	    }
	    (limit * math.pow(2, nzeros*power)).toInt
	  }
	}
	
	def powerLimit(round:Int, limit:Int):Int = powerLimit(round, limit, 1f);
	
	var powerLimitFctn = powerLimit(_:Int,_:Int);
}