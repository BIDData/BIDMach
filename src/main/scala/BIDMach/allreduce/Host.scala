package BIDMach.allreduce

import BIDMat.{Mat,SBMat,CMat,DMat,FMat,IMat,HMat,GDMat,GLMat,GMat,GIMat,GSDMat,GSMat,LMat,SMat,SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import edu.berkeley.bid.comm._
import scala.collection.parallel._
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.net.ServerSocket;
import java.net.Socket;
import java.net.SocketException;
import java.net.InetSocketAddress;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;


import java.io.IOException;


class Host(val opts:Host.Opts = new Host.Options) extends Serializable {
  
  var M = 0;
  var imach = 0;
  var round = 0;
  var gmods:IMat = null;
  var gridmachines:IMat = null;
  var workerIPs:IMat = null; 
  var machineIPs:Array[String] = null;
  var groups:Groups = null;
  var executor:ExecutorService = null;  
  
  def log(msg:String) {
    print(msg); 
  }
 
  class TimeoutThread(mtime:Int, futures:Array[Future[_]]) extends Runnable {   
	  def run() {
		  try {
			  Thread.sleep(mtime);
			  for (i <- 0 until futures.length) {
				  if (futures(i) != null) {
					  if (opts.trace > 0) log("Worker cancelling thread %d" format i);
					  futures(i).cancel(true);
				  }
			  }
		  } catch {
		  case e:InterruptedException => if (opts.trace > 2) log("Worker interrupted timeout thread");
		  }
	  }
  }
}

object Host {
  trait Opts extends BIDMat.Opts{
    var sendTimeout = 1000;
    var recvTimeout = 1000;
    var trace = 0;
    var machineTrace = 0;
    var commandSocketNum = 50050;
    var responseSocketNum = 50049;
    var peerSocketNum = 50051;
  }
  
  class Options extends Opts {} 
  
}

