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


class Master(val opts:Master.Opts = new Master.Options) extends Serializable {
  
  var M = 0;
	var gmods:IMat = null;
	var gridmachines:IMat = null;
	var workerIPs:IMat = null
	var executor:ExecutorService = null;
	var reduceTask:Future[_] = null;
	var reducer:Reducer = null

	
	def init() {
	  executor = Executors.newFixedThreadPool(opts.numThreads);
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
    reducer = new Reducer();
    reduceTask = executor.submit(reducer);
  }
  
  def stopUpdates() {
    reducer.stop = true;
    reduceTask.cancel(true);    
  }
  
  def log(msg:String) {
		print(msg);	
	}
    
  def broadcastCommand(cmd:Command) {
  	cmd.encode;
  	if (opts.trace > 2) log("Broadcasting cmd %s" format cmd);
  	val futures = (0 until M).toArray.map(imach => {
  	  val newcmd = new Command(cmd.ctype, cmd.clen, cmd.bytes);
      newcmd.imach = imach;
      send(newcmd, workerIPs(imach));    
	  });
	}
  
  def send(cmd:Command, address:Int):Future[_] = {
    val cw = new CommandWriter(Command.toAddress(address), opts.socketNum, cmd);
    executor.submit(cw);
  }
  
  class CommandWriter(dest:String, socketnum:Int, command:Command) extends Runnable {

  	def run() {
  		var socket:Socket = null;
  	  try {
  	  	socket = new Socket();
  	  	socket.connect(new InetSocketAddress(dest, socketnum), opts.sendTimeout);
  	  	if (socket.isConnected()) {
  	  		val ostr = new DataOutputStream(socket.getOutputStream());
  	  		ostr.writeInt(command.magic)
  	  		ostr.writeInt(command.ctype);
  	  		ostr.writeInt(command.clen);
  	  		ostr.write(command.bytes, 0, command.clen*4);		
  	  	}
  	  }	catch {
  	  case e:Exception =>
  	  if (opts.trace > 0) {
  	    log("Master problem sending command %s\n%s" format (command.toString, Command.printStackTrace(e)));
  	  }
  	  } finally {
  	  	try { if (socket != null) socket.close(); } catch {
  	  	case e:Exception =>
  	  	if (opts.trace > 0) log("Master problem closing socket\n%s" format Command.printStackTrace(e));			  
  	  	}
  	  }
  	}
  }
  
  class Reducer() extends Runnable {
  	var stop = false;

  	def run() {
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
  		broadcastCommand(cmd);
  		round += 1;
  	}
  }
  
  class TimeoutThread(mtime:Int, futures:Array[Future[_]]) extends Runnable {		
		def run() {
			try {
				Thread.sleep(mtime);
				for (i <- 0 until futures.length) {
					if (futures(i) != null) {
						if (opts.trace > 0) log("Master cancelling thread %d" format i);
						futures(i).cancel(true);
					}
				}
			} catch {
			  case e:InterruptedException => if (opts.trace > 2) log("Master interrupted timeout thread");
			}
		}
  }
}

object Master {
	trait Opts extends BIDMat.Opts{
		var limit = 0;
		var limitFctn:(Int,Int)=>Int = null;
		var intervalMsec = 100;
		var timeScaleMsec = 1e-5f;
		var permuteAlways = true;
		var sendTimeout = 1000;
		var recvTimeout = 1000;
		var trace = 0;
		var socketNum = 50051;
		var numThreads = 16;
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

