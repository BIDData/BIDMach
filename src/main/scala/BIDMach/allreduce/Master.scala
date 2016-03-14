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
import java.io.IOException;


class Master(val opts:Master.Opts) extends Serializable {
  
  var M = 0;
  var sendTimeout = 1000;
	var recvTimeout = 1000;
	var trace = 0;
	var socketNum = 50050;
	var gmods:IMat = null;
	var gridmachines:IMat = null;
	var workerIPs:IMat = null
	var executor:ExecutorService = null;
	var reduceTask:Future[_] = null;
	var reducer:Reducer = null

	
	def init() {
	  executor = Executors.newFixedThreadPool(4);
	}
	
	def readConfig(configDir:String) {
		val clengths = loadIMat(configDir + "dims.imat.lz4");
		val allgmods = loadIMat(configDir + "gmods.imat.lz4");
		val allmachinecodes = loadIMat(configDir + "machines.imat.lz4");
		gmods = allgmods(0->clengths(M-1), M-1);
		gridmachines = allmachinecodes(0->M, M-1);
	}
  
  def config(gmods0:IMat, gridmachines0:IMat, workerIPs0:IMat) {
    gmods = gmods0;
    gridmachines = gridmachines0;
    workerIPs = workerIPs0;
    M = workerIPs.length;
  }
  
  def sendConfig() {
    val clen = 4 + gmods.length + gridmachines.length + workerIPs.length;
    val cmd = new ConfigCommand(clen);
    cmd.gmods = gmods;
    cmd.gridmachines = gridmachines;
    cmd.workerIPs = workerIPs;
    broadcastCommand(cmd);
  }
  
  def permuteNodes(seed:Long) {
    val cmd = new PermuteCommand();
    cmd.seed = seed;
    broadcastCommand(cmd);
  }
  
  def startUpdates() {
    reducer = new Reducer(this);
    reduceTask = executor.submit(reducer);
  }
  
  def stopUpdates() {
    reducer.stop = true;
    reduceTask.cancel(true);    
  }
  
  def log(msg:String) {
		print(msg);	
	}
  
  def toAddress(v:Int):String = {
    val p0 = (v >> 24) & 255;
    val p1 = (v >> 16) & 255;
    val p2 = (v >> 8) & 255;
    val p3 = v & 255;
    "%d.%d.%d.%d" format(p0,p1,p2,p3);
   }
    
  def broadcastCommand(cmd:Command) {
	  for (imach <- 0 until M) {
      cmd.imach = imach;
      cmd.encode;
      send(cmd, workerIPs(imach));      
    }
	}
  
  def send(cmd:Command, address:Int) = {
    val cw = new CommandWriter(this, toAddress(address), socketNum, cmd);
    val fut = executor.submit(cw);
  }
}

object Master {
	trait Opts extends BIDMat.Opts{
		var limit = 0;
		var limitFctn:(Int,Int)=>Int = null;
		var intervalMsec = 100;
		var permuteAlways = true;
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

class Reducer(val me:Master) extends Runnable {
  var stop = false;
  
  def run() {
    val opts = me.opts;
    var round = 0;
    var limit = 0;
    while (!stop) {
      Thread.sleep(opts.intervalMsec);
      val newlimit0 = if (opts.limitFctn != null) {
        opts.limitFctn(round, opts.limit);
      } else {
        opts.limit;
      }
      val newlimit = if (newlimit0 < 0) 2000000000 else newlimit0;
    }
    val cmd = if (opts.permuteAlways) {
      val cmd0 = new PermuteAllreduceCommand();
      cmd0.round = round;
      cmd0.seed = round;
      cmd0.limit = limit;
      cmd0;
    } else {
      val cmd0 = new AllreduceCommand();
      cmd0.round = round;
      cmd0.limit = limit;
      cmd0;
    }
    me.broadcastCommand(cmd);
    round += 1;
  }
  
}

