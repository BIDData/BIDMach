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
  var listener:ResponseListener = null;
	var listenerTask:Future[_] = null;
	var reduceTask:Future[_] = null;
	var reducer:Reducer = null
	var sendTiming = false

	
	def init() {
	  executor = Executors.newFixedThreadPool(opts.numThreads);
	  listener = new ResponseListener(opts.responseSocketNum);
	  listenerTask = executor.submit(listener);
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
    val clen = 3 + gmods.length + gridmachines.length + workerIPs.length;
    val cmd = new ConfigCommand(clen, 0);
    cmd.gmods = gmods;
    cmd.gridmachines = gridmachines;
    cmd.workerIPs = workerIPs;
    broadcastCommand(cmd);
  }
  
  def permuteNodes(seed:Long) {
    val cmd = new PermuteCommand(0);
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
  
  def startLearners() {
    val cmd = new StartLearnerCommand(0);
    broadcastCommand(cmd);
  }
  
  def permuteAllreduce(round:Int, limit:Int) {
  	val cmd = new PermuteAllreduceCommand(0);
  	cmd.round = round;
  	cmd.seed = round;
  	cmd.limit = limit;
  	broadcastCommand(cmd);
  }
  
  def log(msg:String) {
		print(msg);	
	}
    
  def broadcastCommand(cmd:Command) {
  	cmd.encode;
  	if (opts.trace > 2) log("Broadcasting cmd %s\n" format cmd);
  	val futures = new Array[Future[_]](M);
  	sendTiming = true;
  	val timeout = executor.submit(new TimeoutThread(opts.sendTimeout, futures));
  	for (imach <- 0 until M) {
  	  val newcmd = new Command(cmd.ctype, imach, cmd.clen, cmd.bytes);
  		futures(imach) = send(newcmd, workerIPs(imach));   
  	}
  	for (imach <- 0 until M) {
  		try {
  			futures(imach).get() 
  		} catch {
  		case e:Exception => {}
  		}
  		if (futures(imach).isCancelled()) {
  			if (opts.trace > 0) log("Broadcast to machine %d timed out, cmd %s\n" format (imach, cmd));
  		}
  	}
  	sendTiming = false;
  	timeout.cancel(true);
  }
  
  def setMachineNumbers {
  	if (opts.trace > 2) log("Broadcasting setMachineNumbers\n");
  	val futures = new Array[Future[_]](M);
  	sendTiming = true;
  	val timeout = executor.submit(new TimeoutThread(opts.sendTimeout, futures));
  	for (imach <- 0 until M) {
  	  val cmd = new SetMachineCommand(0, imach);
  	  cmd.encode
  		futures(imach) = send(cmd, workerIPs(imach));   
  	}
  	for (imach <- 0 until M) {
  		try {
  			futures(imach).get() 
  		} catch {
  		case e:Exception => {}
  		}
  		if (futures(imach).isCancelled()) {
  			if (opts.trace > 0) log("Broadcast to machine %d timed out, cmd setMachineNumbers\n" format (imach));
  		}
  	}
  	sendTiming = false;
  	timeout.cancel(true);
  }
  
  def send(cmd:Command, address:Int):Future[_] = {
    val cw = new CommandWriter(Command.toAddress(address), opts.commandSocketNum, cmd);
    executor.submit(cw);
  }
  
  class CommandWriter(dest:String, socketnum:Int, command:Command) extends Runnable {

  	def run() {
  		var socket:Socket = null;
  	  try {
  	  	socket = new Socket();
  	  	socket.setReuseAddress(true);
  	  	socket.connect(new InetSocketAddress(dest, socketnum), opts.sendTimeout);
  	  	if (socket.isConnected()) {
  	  		val ostr = new DataOutputStream(socket.getOutputStream());
  	  		ostr.writeInt(command.magic)
  	  		ostr.writeInt(command.ctype);
  	  		ostr.writeInt(command.dest);
  	  		ostr.writeInt(command.clen);
  	  		ostr.write(command.bytes, 0, command.clen*4);		
  	  	}
  	  }	catch {
  	  case e:Exception =>
  	  if (opts.trace > 0) {
  	    log("Master problem sending command %s\n%s\n" format (command.toString, Command.printStackTrace(e)));
  	  }
  	  } finally {
  	  	try { if (socket != null) socket.close(); } catch {
  	  	case e:Exception =>
  	  	if (opts.trace > 0) log("Master problem closing socket\n%s\n" format Command.printStackTrace(e));			  
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
  			val newlimit0 = if (opts.limitFctn != null) {
  				opts.limitFctn(round, opts.limit);
  			} else {
  				opts.limit;
  			}
  			limit = if (newlimit0 <= 0) 2000000000 else newlimit0;
  			val cmd = if (opts.permuteAlways) {
  				val cmd0 = new PermuteAllreduceCommand(0);
  				cmd0.round = round;
  				cmd0.seed = round;
  				cmd0.limit = limit;
  				cmd0;
  			} else {
  				val cmd0 = new AllreduceCommand(0);
  				cmd0.round = round;
  				cmd0.limit = limit;
  				cmd0;
  			}
  			broadcastCommand(cmd);
  			val timems = opts.intervalMsec + (limit * opts.timeScaleMsec).toInt;
  			if (opts.trace > 2) log("Sleeping for %d msec\n" format timems);
  			Thread.sleep(timems);
  			round += 1;
  		}
  	}
  }
  
  class TimeoutThread(mtime:Int, futures:Array[Future[_]]) extends Runnable {		
		def run() {
			try {
				Thread.sleep(mtime);
				if (sendTiming) {
					for (i <- 0 until futures.length) {
						if (futures(i) != null) {
							if (opts.trace > 0) log("Master cancelling thread %d\n" format i);
							futures(i).cancel(true);
						}
					}
				}
			} catch {
			  case e:InterruptedException => if (opts.trace > 3) log("Master interrupted timeout thread %s\n" format Command.printStackTrace(e));
			}
		}
  }
  
    
  def stop = {
    listener.stop = true;
    listenerTask.cancel(true);
  }
  
  def shutdown = {
    executor.shutdownNow();
    val tt= toc;
  }
  
  def handleResponse(response:Response) = {
    if (response.magic != Response.magic) {
      if (opts.trace > 0) log("Master got message with bad magic number %d\n" format (response.magic));      
    } else {
    	response.rtype match {
    	case Response.configRtype => {
    		
    	}
    	}
    }
  }

	class ResponseListener(val socketnum:Int) extends Runnable {
		var stop = false;
		var ss:ServerSocket = null;

		def start() {
			try {
				ss = new ServerSocket(socketnum);
			} catch {
			case e:Exception => {if (opts.trace > 0) log("Problem in ResponseListener\n%s" format Response.printStackTrace(e));}
			}
		}

		def run() {
			start();
			while (!stop) {
				try {
					val scs = new ResponseReader(ss.accept());
					if (opts.trace > 2) log("Command Listener got a message\n");
					val fut = executor.submit(scs);
				} catch {
				case e:SocketException => {
				  if (opts.trace > 0) log("Problem starting a socket reader\n%s" format Response.printStackTrace(e));
				}
				// This is probably due to the server shutting to. Don't do anything.
				case e:Exception => {
					if (opts.trace > 0) log("Master Response listener had a problem "+e);
				}
				}
			}
		}

		def stop(force:Boolean) {
			stop = true;
			if (force) {
				try {
					stop = true;
					ss.close();
				} catch {
				case e:Exception => {
					if (opts.trace > 0) log("Master trouble closing response listener\n%s" format ( Response.printStackTrace(e)));
				}
				}			
			}
		}
	}

	class ResponseReader(socket:Socket) extends Runnable {
		def run() {
			try {
				val istr = new DataInputStream(socket.getInputStream());
				val magic = istr.readInt();
				val ctype = istr.readInt();
				val dest = istr.readInt();
				val clen = istr.readInt();
				val response = new Response(ctype, dest, clen, new Array[Byte](clen*4));
				if (opts.trace > 2) log("Master got packet %s\n" format (response.toString));
				istr.readFully(response.bytes, 0, clen*4);
				try {
  				socket.close();
				} catch {
				case e:IOException => {if (opts.trace > 0) log("Master Problem closing socket "+Response.printStackTrace(e)+"\n")}
				}
				handleResponse(response);
			} catch {
			case e:Exception =>	if (opts.trace > 0) log("Master Problem reading socket "+Response.printStackTrace(e)+"\n");
			} finally {
				try {
					if (!socket.isClosed) socket.close();
				} catch {
				case e:IOException => {if (opts.trace > 0) log("Master Final Problem closing socket "+Response.printStackTrace(e)+"\n")}
				}
			}
		}
	}

}

object Master {
	trait Opts extends BIDMat.Opts{
		var limit = 0;
		var limitFctn:(Int,Int)=>Int = null;
		var intervalMsec = 1000;
		var timeScaleMsec = 1e-4f;
		var permuteAlways = true;
		var sendTimeout = 1000;
		var recvTimeout = 1000;
		var trace = 0;
		var commandSocketNum = 50050;
		var responseSocketNum = 50049;
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

