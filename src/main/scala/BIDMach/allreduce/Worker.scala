package BIDMach.allreduce

import BIDMat.{Mat,SBMat,CMat,DMat,FMat,IMat,HMat,GDMat,GLMat,GMat,GIMat,GSDMat,GSMat,LMat,SMat,SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import BIDMach.Learner;
import BIDMach.models.Model;
import edu.berkeley.bid.comm._
import scala.collection.parallel._
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.net.ServerSocket;
import java.net.Socket;
import java.net.SocketException;
import java.io.DataInputStream;
import java.io.IOException;

class Worker(val opts:Worker.Opts = new Worker.Options) extends Serializable {

  var M = 0;
  var imach = 0;
	var gmods:IMat = null;
	var gridmachines:IMat = null;  
	var machineIPs:Array[String] = null;
	var groups:Groups = null;

	var executor:ExecutorService = null;
	var listener:CommandListener = null;
	var listenerTask:Future[_] = null;
	var machine:Machine = null;
	var learner:Learner = null;
	var model:Model = null;
	
	def start(learner0:Learner) = {
	  learner = learner0;
	  if (model == null && learner != null) model = learner.model;
	  executor = Executors.newFixedThreadPool(8);
	  listener = new CommandListener(opts.commandSocketNum);
	  listenerTask = executor.submit(listener);
	}
  
  def config(imach0:Int, gmods0:IMat, gridmachines0:IMat, machineIPs0:IMat) = {    
    imach = imach0;
    gmods = gmods0;
    gridmachines = gridmachines0;
    M = gridmachines.length;
    groups = new Groups(M, gmods.data, gridmachines.data, 0);
    machineIPs = machineIPs0.data.map(Command.toAddress(_));
    if (machine != null) machine.stop;
    machine = new Machine(null, groups, imach, M, opts.useLong, opts.bufsize, false, opts.trace, opts.replicate, machineIPs);
    machine.configTimeout = opts.configTimeout;
    machine.reduceTimeout = opts.reduceTimeout;
    machine.sendTimeout = opts.sendTimeout;
    machine.recvTimeout = opts.recvTimeout;
    machine.sockBase = opts.peerSocketNum;
    machine.sockOffset = 0;
    machine.start(machine.maxk);
  }
  
  def permute(seed:Long) = {
    machine.groups.permute(seed.toInt);
  }
  
  def allReduce(round:Int, limit:Long) = {
    if (model != null) {
    	model.snapshot(limit.toInt, opts.doAvg);
    	val sendmat = model.sendmat;
    	val recvmat = model.recvmat;
    	val indexmat = if (model.indexmat.asInstanceOf[AnyRef] != null) {
    		model.indexmat
    	} else {
    		irow(0 -> sendmat.ncols)
    	}

    	val result = if (opts.fuseConfigReduce) {
    		(indexmat, sendmat) match {
    		case (lmat:LMat, fsendmat:FMat) =>  machine.configReduce(lmat.data, lmat.data, fsendmat.data, sendmat.nrows, round);
    		case (imat:IMat, fsendmat:FMat) =>  machine.configReduce(imat.data, imat.data, fsendmat.data, sendmat.nrows, round);
    		}
    	} else {
    		(indexmat, sendmat) match {
    		case (lmat:LMat, fsendmat:FMat) =>  machine.config(lmat.data, lmat.data, round);
    		case (imat:IMat, fsendmat:FMat) =>  machine.config(imat.data, imat.data, round);
    		}
    		machine.reduce(sendmat.asInstanceOf[FMat].data, sendmat.nrows, round);
    	}
    	model.addStep(limit.toInt, opts.doAvg);
    } else {
      if (opts.trace > 2) log("Allreduce model is null\n")
    }
	}
  
  def stop = {
    listener.stop = true;
    listenerTask.cancel(true);
    machine.stop;
  }
  
  def shutdown = {
    executor.shutdownNow();
    val tt= toc;
  }
  
  def handleCMD(cmd:Command) = {
    if (cmd.magic != Command.magic) {
      if (opts.trace > 0) log("Machine %d got message with bad magic number %d\n" format (imach, cmd.magic));
    }  else if (cmd.dest != imach) {
      	if (opts.trace > 0) log("Machine %d got message with bad destination %d\n" format (imach, cmd.dest));
    } else {
    	cmd.ctype match {
    	case Command.configCtype => {
    		val newcmd = new ConfigCommand(cmd.clen, imach, cmd.bytes);
    		newcmd.decode;
    		if (opts.trace > 2) log("Received %s\n" format newcmd.toString);
    		config(newcmd.dest, newcmd.gmods, newcmd.gridmachines, newcmd.workerIPs);
    	}
    	case Command.permuteCtype => {
    		val newcmd = new PermuteCommand(cmd.dest, cmd.bytes);
    		newcmd.decode;
    		if (opts.trace > 2) log("Received %s\n" format newcmd.toString);
    		permute(newcmd.seed);
    	}
    	case Command.allreduceCtype => {
    		val newcmd = new AllreduceCommand(cmd.dest, cmd.bytes);
    		newcmd.decode;
    		if (opts.trace > 2) log("Received %s\n" format newcmd.toString);
    		allReduce(newcmd.round, newcmd.limit);
    	}
    	case Command.permuteAllreduceCtype => {
    		val newcmd = new PermuteAllreduceCommand(cmd.dest, cmd.bytes);
    		newcmd.decode;
    		if (opts.trace > 2) log("Received %s\n" format newcmd.toString);
    		permute(newcmd.seed);
    		allReduce(newcmd.round, newcmd.limit);
    	}
    	case Command.setMachineCtype => {
    		val newcmd = new SetMachineCommand(cmd.dest, cmd.bytes);
    		newcmd.decode;
    		if (opts.trace > 2) log("Received %s\n" format newcmd.toString);
    		imach = newcmd.dest;
    	}
    	case Command.startLearnerCtype => {
    		val newcmd = new StartLearnerCommand(cmd.dest, cmd.bytes);
    		newcmd.decode;
    		if (opts.trace > 2) log("Received %s\n" format newcmd.toString);
    		if (learner != null) {
    		  learner.paused = false;
    		}
    	}
    	}
    }
  }

	class CommandListener(val socketnum:Int) extends Runnable {
		var stop = false;
		var ss:ServerSocket = null;

		def start() {
			try {
				ss = new ServerSocket(socketnum);
			} catch {
			case e:Exception => {if (opts.trace > 0) log("Problem in CommandListener\n%s" format Command.printStackTrace(e));}
			}
		}

		def run() {
			start();
			while (!stop) {
				try {
					val scs = new CommandReader(ss.accept());
					if (opts.trace > 2) log("Command Listener got a message\n");
					val fut = executor.submit(scs);
				} catch {
				case e:SocketException => {
				  if (opts.trace > 0) log("Problem starting a socket reader\n%s" format Command.printStackTrace(e));
				}
				// This is probably due to the server shutting to. Don't do anything.
				case e:Exception => {
					if (opts.trace > 0) log("Machine %d Command listener had a problem "+e format imach);
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
					if (opts.trace > 0) log("Machine %d trouble closing command listener\n%s" format (imach, Command.printStackTrace(e)));
				}
				}			
			}
		}
	}

	class CommandReader(socket:Socket) extends Runnable {
		def run() {
			try {
				val istr = new DataInputStream(socket.getInputStream());
				val magic = istr.readInt();
				val ctype = istr.readInt();
				val dest = istr.readInt();
				val clen = istr.readInt();
				val cmd = new Command(ctype, dest, clen, new Array[Byte](clen*4));
				if (opts.trace > 2) log("Worker %d got packet %s\n" format (imach, cmd.toString));
				istr.readFully(cmd.bytes, 0, clen*4);
				try {
  				socket.close();
				} catch {
				case e:IOException => {if (opts.trace > 0) log("Worker %d Problem closing socket "+e.toString()+"\n" format (imach))}
				}
				handleCMD(cmd);
			} catch {
			case e:Exception =>	if (opts.trace > 0) log("Worker %d Problem reading socket "+e.toString()+"\n" format (imach));
			} finally {
				try {
					if (!socket.isClosed) socket.close();
				} catch {
				case e:IOException => {if (opts.trace > 0) log("Worker %d Final Problem closing socket "+e.toString()+"\n" format (imach))}
				}
			}
		}
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
  
  def log(msg:String) {
		print(msg);	
	}
}

object Worker {
	trait Opts extends BIDMat.Opts{
		var configTimeout = 3000;
		var reduceTimeout = 3000;
		var sendTimeout = 1000;
		var recvTimeout = 1000;
		var cmdTimeout = 1000;
		var commandSocketNum = 50050;
		var peerSocketNum = 50051;
		var fuseConfigReduce = false;
		var doAvg = true;
		var useLong = false;
		var trace = 0;
		var replicate = 1;
		var bufsize = 10*1000000;
  }
	
	class Options extends Opts {}
}