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
  var replicate = 1;
	var gmods:IMat = null;
	var gridmachines:IMat = null;  
	var machineIPs:Array[String] = null;
	var groups:Groups = null;
	var socketNum = 50050;

	var executor:ExecutorService = null;
	var listener:CommandListener = null;
	var listenerTask:Future[_] = null;
	var machine:Machine = null;
	var learner:Learner = null;
	var model:Model = null;
	var bufsize = 10*1000000;
	
	def start(learner0:Learner) = {
	  learner = learner0;
	  model = learner.model;
	  executor = Executors.newFixedThreadPool(4);
	  listener = new CommandListener(this, socketNum);
	  listenerTask = executor.submit(listener);
	}

  
  def config(imach0:Int, gmods0:IMat, gridmachines0:IMat, machineIPs0:IMat) = {    
    imach = imach0;
    gmods = gmods0;
    gridmachines = gridmachines0;
    M = gridmachines.length;
    groups = new Groups(M, gmods.data, gridmachines.data, 0);
    machineIPs = machineIPs0.data.map(Command.toAddress(_));
    
    var totvals = 0L;

    machine = new Machine(null, groups, imach, M, opts.useLong, bufsize, false, opts.trace, replicate, machineIPs);
    machine.configTimeout = opts.configTimeout;
    machine.reduceTimeout = opts.reduceTimeout;
    machine.sendTimeout = opts.sendTimeout;
    machine.recvTimeout = opts.recvTimeout;
  }
  
  def allReduce(round:Int, limit:Int) = {
    model.snapshot(limit, opts.doAvg);
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
    model.addStep(limit, opts.doAvg)
	}
  
  def stop = {
    listener.stop = true;
    listenerTask.cancel(true);
  }
  
  def shutdown = {
    executor.shutdownNow();
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
		var doAvg = true;
		var useLong = false;
		var trace = 0;
  }
	
	class Options extends Opts {}
}