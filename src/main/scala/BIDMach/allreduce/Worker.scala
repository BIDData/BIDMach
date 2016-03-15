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

class Worker(val opts:Worker.Opts) extends Serializable {

  var M = 0;
  var imach = 0;
	var gmods:IMat = null;
	var gridmachines:IMat = null;  
	var machineIPs:Array[String] = null;
	var groups:Groups = null;
	var socketNum = 50051;

	var executor:ExecutorService = null;
	var listener:CommandListener = null;
	var listenerTask:Future[_] = null;
	var machine:Machine = null;
	var learner:Learner = null;
	var model:Model = null;
	
	def start(learner0:Learner) = {
	  learner = learner0;
	  model = learner.model;
	  executor = Executors.newFixedThreadPool(4);
	  listener = new CommandListener(socketNum);
	  listenerTask = executor.submit(listener);
	}
  
  def config(imach0:Int, gmods0:IMat, gridmachines0:IMat, machineIPs0:IMat) = {    
    imach = imach0;
    gmods = gmods0;
    gridmachines = gridmachines0;
    M = gridmachines.length;
    groups = new Groups(M, gmods.data, gridmachines.data, 0);
    machineIPs = machineIPs0.data.map(Command.toAddress(_));

    machine = new Machine(null, groups, imach, M, opts.useLong, opts.bufsize, false, opts.trace, opts.replicate, machineIPs);
    machine.configTimeout = opts.configTimeout;
    machine.reduceTimeout = opts.reduceTimeout;
    machine.sendTimeout = opts.sendTimeout;
    machine.recvTimeout = opts.recvTimeout;
  }
  
  def permute(seed:Long) = {
    machine.groups.permute(seed.toInt);
  }
  
  def allReduce(round:Int, limit:Long) = {
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
    model.addStep(limit.toInt, opts.doAvg)
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
    cmd.ctype match {
      case Command.configCtype => {
        val newcmd = new ConfigCommand(cmd.clen, cmd.bytes);
        newcmd.decode;
        config(newcmd.imach, newcmd.gmods, newcmd.gridmachines, newcmd.workerIPs);
      }
      case Command.permuteCtype => {
        val newcmd = new PermuteCommand(cmd.bytes);
        newcmd.decode;
        permute(newcmd.seed);
      }
      case Command.allreduceCtype => {
        val newcmd = new AllreduceCommand(cmd.bytes);
        newcmd.decode;
        allReduce(newcmd.round, newcmd.limit);
      }
      case Command.permuteAllreduceCtype => {
        val newcmd = new PermuteAllreduceCommand(cmd.bytes);
        newcmd.decode;
        permute(newcmd.seed);
        allReduce(newcmd.round, newcmd.limit);
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
			case e:Exception => {}
			}
		}

		def run() {
			start();
			while (!stop) {
				try {
					val scs = new CommandReader(ss.accept());
					val fut = executor.submit(scs);
				} catch {
				case e:SocketException => {
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
					if (opts.trace > 0) log("Machine %d trouble closing command listener" format imach);
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
				val clen = istr.readInt();
				val cmd = new Command(ctype, clen, new Array[Byte](clen*4));
				if (opts.trace > 2) log("Worker %d got packet %s\n" format (imach, cmd.toString));
				var waiting = 0;
				while (waiting < opts.cmdTimeout) {
					Thread.sleep(10);
					waiting += 10;
				}
				istr.readFully(cmd.bytes, 0, clen*4);
				handleCMD(cmd);
			} catch {
			case e:Exception =>	if (opts.trace > 0) log("Worker %d Problem reading socket "+e.toString()+"\n" format (imach));
			} finally {
				try {
					socket.close();
				} catch {
				case e:IOException => {}
				}
			}
		}
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
		var doAvg = true;
		var useLong = false;
		var trace = 0;
		var replicate = 1;
		var bufsize = 10*1000000;
  }
	
	class Options extends Opts {}
}