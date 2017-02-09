package BIDMach.allreduce

import BIDMat.{Mat, SBMat, CMat, DMat, FMat, IMat, HMat, GDMat, GLMat, GMat, GIMat, GSDMat, GSMat, LMat, SMat, SDMat}
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
import java.net.InetAddress;
import java.net.InetSocketAddress;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import javax.script.ScriptEngine;
import javax.script.ScriptEngineManager;
import java.util.concurrent.Callable
import javax.script.ScriptContext

class Worker(override val opts: Worker.Opts = new Worker.Options) extends Host {

  var listener:CommandListener = null;
  var listenerTask:Future[_] = null;
  var machine:Machine = null;
  var learner:Learner = null;
  var obj:AnyRef = null;
  var str:String = null;
  var model:Model = null;
  var intp:ScriptEngine = null;
  var masterSocketAddr:InetSocketAddress = null;
  var workerIP:InetAddress = null;

  def start(learner0:Learner) = {
    workerIP = InetAddress.getLocalHost;
    learner = learner0;
    if (model == null && learner != null) model = learner.model;
    executor = Executors.newFixedThreadPool(8);
    listener = new CommandListener(opts.commandSocketNum, this);
    listenerTask = executor.submit(listener);
    intp = new ScriptEngineManager().getEngineByName("scala");
    intp.put("worker", "try");
    intp.put("worker2", this);
  }

  def config(imach0: Int, gmods0: IMat, gridmachines0: IMat, workers0: Array[InetSocketAddress],
             masterSocketAddr0: InetSocketAddress) = {
    val t1 = toc;
    imach = imach0;
    gmods = gmods0;
    gridmachines = gridmachines0;
    M = gridmachines.length;
    groups = new Groups(M, gmods.data, gridmachines.data, 0);
    workers = workers0;
    masterSocketAddr = masterSocketAddr0;
    if (machine != null) machine.stop;
    machine = new Machine(null, groups, imach, M, opts.useLong, opts.bufsize, false, opts.machineTrace, opts.replicate, workers);
    machine.configTimeout = opts.configTimeout;
    machine.reduceTimeout = opts.reduceTimeout;
    machine.sendTimeout = opts.sendTimeout;
    machine.recvTimeout = opts.recvTimeout;
    machine.sockBase = opts.peerSocketNum;
    machine.start(machine.maxk);
    intp.put("$imach", imach);
    val t2 = toc
    if (opts.trace > 2) log("Machine config took %4.3f secs\n" format (t2 - t1))
  }

  def permute(seed: Long) = {
    machine.groups.permute(seed.toInt);
  }

  def allReduce(round: Int, limit: Long) = {
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
        case im: IMat => math.min(limit, im.length) * (2 + 2 * sendmat.nrows) * 8f;
        case im: LMat => math.min(limit, im.length) * (4 + 2 * sendmat.nrows) * 8f;
      }
      if (opts.trace > 2) log("Allreduce %5.2f MB took %5.4f secs at %5.2f MB/sec\n" format(nbytes / 1e6f, t2 - t1, nbytes / (t2 - t1) / 1e6f))
    } else {
      if (opts.trace > 2) log("Allreduce model is null\n")
    }
  }

  def stop = {
    listener.stop = true;
    listenerTask.cancel(true);
    if (machine != null) machine.stop;
  }

  def shutdown = {
    executor.shutdownNow();
    val tt = toc;
  }

  def handleCMD(cmd: Command) = {
    if (cmd.magic != Command.magic) {
      if (opts.trace > 0) log("Machine %d got message with bad magic number %d\n" format (imach, cmd.magic));
      }  else if (cmd.dest != imach) {
	if (opts.trace > 0) log("Machine %d got message with bad destination %d\n" format (imach, cmd.dest));
	} else {
	  cmd.ctype match {
	    case Command.configCtype => {
	      val newcmd = new ConfigCommand(0, 0, null, null, null, null, -1, -1, cmd.clen, cmd.bytes);
	      newcmd.decode;
	      if (opts.trace > 2) log("Received %s\n" format newcmd.toString);
	      config(newcmd.dest, newcmd.gmods, newcmd.gridmachines,
		Host.hostPortsToInet(newcmd.workerIPs, newcmd.workerPorts),
		Host.hostPortToInet(newcmd.masterIP, newcmd.masterResPort)
		);
	      if (opts.respond > 0) sendMaster(new Response(Command.configCtype, newcmd.round, imach));
	    }
	    case Command.permuteCtype => {
	      val newcmd = new PermuteCommand(0, cmd.dest, 0, cmd.bytes);
	      newcmd.decode;
	      if (opts.trace > 2) log("Received %s\n" format newcmd.toString);
	      permute(newcmd.seed);
	      if (opts.respond > 0) sendMaster(new Response(Command.permuteCtype, newcmd.round, imach));
	    }
	    case Command.allreduceCtype => {
	      val newcmd = new AllreduceCommand(0, cmd.dest, 0, cmd.bytes);
	      newcmd.decode;
	      if (opts.trace > 2) log("Received %s\n" format newcmd.toString);
	      allReduce(newcmd.round, newcmd.limit);
	    if (opts.respond > 0) sendMaster(new Response(Command.allreduceCtype, newcmd.round, imach));
	    }
	    case Command.permuteAllreduceCtype => {
	      val newcmd = new PermuteAllreduceCommand(0, cmd.dest, 0, 0, cmd.bytes);
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
	      val newcmd = new SendLearnerCommand(0, cmd.dest, null, cmd.bytes);
	      newcmd.decode;
	      learner = newcmd.learner;
	      if (opts.trace > 2) log("Received %s\n" format newcmd.toString);
	      if (opts.respond > 0) sendMaster(new Response(Command.sendLearnerCtype, newcmd.round, imach));
	    }
	    case Command.assignObjectCtype => {
	      val newcmd = new AssignObjectCommand(0, cmd.dest, null, null, cmd.bytes);
	      newcmd.decode;
	      obj = newcmd.obj;
	      str = newcmd.str;
	      intp.put(str, obj);
	      //intp.getBindings(ScriptContext.ENGINE_SCOPE).put(str,obj);
	      if (opts.trace > 2) log("Received %s\n" format newcmd.toString);
	      if (opts.respond > 0) sendMaster(new Response(Command.assignObjectCtype, newcmd.round, imach));
	    }
	    case Command.evalStringCtype => {
	      val newcmd = new EvalStringCommand(0, cmd.dest, null, cmd.bytes);
	      newcmd.decode;
	      str = newcmd.str;
	      obj = intp.eval(str);
	      val resp = new ReturnObjectResponse(cmd.round, cmd.dest, obj);
	      sendMaster(resp);
	      if (opts.trace > 2) log("Received %s\n" format newcmd.toString);
	    }
	    case Command.callCtype => {
	      val newcmd = new CallCommand(0, cmd.dest, null, cmd.bytes);
	      newcmd.decode;
	      obj = newcmd.func();
	      if (opts.trace > 2) log("Received %s\n" format newcmd.toString);
	      if (opts.trace > 3) log("Computed %s\n" format obj.toString);
	      val resp = new ReturnObjectResponse(cmd.round, cmd.dest, obj);
	      sendMaster(resp);
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
    val cw = new ResponseWriter(masterSocketAddr, resp, this);
    executor.submit(cw);
  }
}

object Worker {
<<<<<<< HEAD

  trait Opts extends Host.Opts {
    var configTimeout = 3000;
    var reduceTimeout = 3000;
    var cmdTimeout = 1000;
    var fuseConfigReduce = false;
    var doAvg = true;
    var useLong = false;
    var replicate = 1;
    var bufsize = 10 * 1000000;
    var respond = 0;
  }

  class Options extends Opts {}

}
