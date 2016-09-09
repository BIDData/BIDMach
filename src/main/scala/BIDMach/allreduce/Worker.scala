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
import java.net.InetSocketAddress;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import scala.tools.nsc.interpreter.IMain

class Worker(override val opts:Worker.Opts = new Worker.Options) extends Host {

	var listener:CommandListener = null;
	var listenerTask:Future[_] = null;
	var machine:Machine = null;
	var learner:Learner = null;
	var obj:AnyRef = null;
	var model:Model = null;
	var master:Int = 0;
  var intp:IMain = null;
	
	def start(learner0:Learner) = {
	  learner = learner0;
	  if (model == null && learner != null) model = learner.model;
	  executor = Executors.newFixedThreadPool(8);
	  listener = new CommandListener(opts.commandSocketNum, this);
	  listenerTask = executor.submit(listener);
	}
  
  def config(imach0:Int, gmods0:IMat, gridmachines0:IMat, machineIPs0:IMat) = {
    val t1 = toc;
    imach = imach0;
    gmods = gmods0;
    gridmachines = gridmachines0;
    M = gridmachines.length;
    groups = new Groups(M, gmods.data, gridmachines.data, 0);
    machineIPs = machineIPs0.data.map(Command.toAddress(_));
    if (machine != null) machine.stop;
    machine = new Machine(null, groups, imach, M, opts.useLong, opts.bufsize, false, opts.machineTrace, opts.replicate, machineIPs);
    machine.configTimeout = opts.configTimeout;
    machine.reduceTimeout = opts.reduceTimeout;
    machine.sendTimeout = opts.sendTimeout;
    machine.recvTimeout = opts.recvTimeout;
    machine.sockBase = opts.peerSocketNum;
    machine.sockOffset = 0;
    machine.start(machine.maxk);
    val t2 = toc
    if (opts.trace > 2) log("Machine config took %4.3f secs\n" format(t2-t1))
  }
  
  def permute(seed:Long) = {
    machine.groups.permute(seed.toInt);
  }
  
  def allReduce(round:Int, limit:Long) = {
    if (model != null) {
      val t1=toc;
    	model.snapshot(limit.toInt, opts.doAvg);
    	val sendmat = model.sendmat;
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
    	model.recvmat = new FMat(sendmat.nrows, sendmat.ncols, result);
    	model.addStep(limit.toInt, opts.doAvg);
    	val t2 = toc;
    	val nbytes = indexmat match {
    	  case im:IMat => math.min(limit, im.length)*(2 + 2*sendmat.nrows)*8f;
    	  case im:LMat => math.min(limit, im.length)*(4 + 2*sendmat.nrows)*8f;
    	}	
    	if (opts.trace > 2) log("Allreduce %5.2f MB took %5.4f secs at %5.2f MB/sec\n" format (nbytes/1e6f, t2-t1, nbytes/(t2-t1)/1e6f))
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
    		val newcmd = new ConfigCommand(0, 0, cmd.clen, cmd.bytes);
    		newcmd.decode;
    		if (opts.trace > 2) log("Received %s\n" format newcmd.toString);
    		config(newcmd.dest, newcmd.gmods, newcmd.gridmachines, newcmd.workerIPs);
    		if (opts.respond > 0) sendMaster(new Response(Command.configCtype, newcmd.round, imach));
    	}
    	case Command.permuteCtype => {
    		val newcmd = new PermuteCommand(0, cmd.dest, cmd.bytes);
    		newcmd.decode;
    		if (opts.trace > 2) log("Received %s\n" format newcmd.toString);
    		permute(newcmd.seed);
    		if (opts.respond > 0) sendMaster(new Response(Command.permuteCtype, newcmd.round, imach));
    	}
    	case Command.allreduceCtype => {
    		val newcmd = new AllreduceCommand(0, cmd.dest, cmd.bytes);
    		newcmd.decode;
    		if (opts.trace > 2) log("Received %s\n" format newcmd.toString);
    		allReduce(newcmd.round, newcmd.limit);
    		if (opts.respond > 0) sendMaster(new Response(Command.allreduceCtype, newcmd.round, imach));
    	}
    	case Command.permuteAllreduceCtype => {
    		val newcmd = new PermuteAllreduceCommand(0, cmd.dest, cmd.bytes);
    		newcmd.decode;
    		if (opts.trace > 2) log("Received %s\n" format newcmd.toString);
    		permute(newcmd.seed);
    		allReduce(newcmd.round, newcmd.limit);
    		if (opts.respond > 0) sendMaster(new Response(Command.permuteAllreduceCtype, newcmd.round, imach));
    	}
    	case Command.setMachineCtype => {
    		val newcmd = new SetMachineCommand(0, cmd.dest, 0, cmd.bytes);
    		newcmd.decode;
    		if (opts.trace > 2) log("Received %s\n" format newcmd.toString);
    		imach = newcmd.newdest;
    		if (opts.respond > 0) sendMaster(new Response(Command.setMachineCtype, newcmd.round, imach));
    	}
    	case Command.startLearnerCtype => {
    		val newcmd = new StartLearnerCommand(0, cmd.dest, cmd.bytes);
    		newcmd.decode;
    		if (opts.trace > 2) log("Received %s\n" format newcmd.toString);
    		if (learner != null) {
    		  learner.paused = false;
    		}
    		if (opts.respond > 0) sendMaster(new Response(Command.startLearnerCtype, newcmd.round, imach));
    	}
    	case Command.sendLearnerCtype => {
    		val newcmd = new SendObjectCommand(0, cmd.dest, cmd.bytes);
    		newcmd.decode;
    		learner = newcmd.obj.asInstanceOf[Learner];
    		if (opts.trace > 2) log("Received %s\n" format newcmd.toString);
    		if (opts.respond > 0) sendMaster(new Response(Command.sendLearnerCtype, newcmd.round, imach));
    	}
    	case Command.sendObjectCtype => {
    		val newcmd = new SendObjectCommand(0, cmd.dest, cmd.bytes);
    		newcmd.decode;
    		obj = newcmd.obj;
    		if (opts.trace > 2) log("Received %s\n" format newcmd.toString);
    		if (opts.respond > 0) sendMaster(new Response(Command.sendObjectCtype, newcmd.round, imach));
    	}
    	}
    }
  }

	class CommandListener(val socketnum:Int, worker:Worker) extends Runnable {
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
					val scs = new CommandReader(ss.accept(), worker);
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

	
	def sendMaster(resp:Response):Future[_] = {
    val cw = new ResponseWriter(Command.toAddress(master), opts.responseSocketNum, resp, this);
    executor.submit(cw);
  }
}

object Worker {
	trait Opts extends Host.Opts{
		var configTimeout = 3000;
		var reduceTimeout = 3000;
		var cmdTimeout = 1000;
		var fuseConfigReduce = false;
		var doAvg = true;
		var useLong = false;
		var replicate = 1;
		var bufsize = 10*1000000;
		var respond = 0;
  }
	
	class Options extends Opts {}
}