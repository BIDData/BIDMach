package BIDMach.allreduce

import BIDMat.{Mat,SBMat,CMat,DMat,FMat,IMat,HMat,GDMat,GLMat,GMat,GIMat,GSDMat,GSMat,LMat,SMat,SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import BIDMat.JSON
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
import java.util.Arrays

class Worker(override val opts:Worker.Opts = new Worker.Options) extends Host {

	var listener:CommandListener = null;
	var listenerTask:Future[_] = null;
	var machineArr:Array[Machine] = null;
	var learner:Learner = null;
	var obj:AnyRef = null;
	var str:String = null;
	var model:Model = null;
	var intp:ScriptEngine = null;
	var masterSocketAddr:InetSocketAddress = null;
	var workerIP:InetAddress = null;
	var lastSoftmax:FMat = null
	var lastScore:Double = -10.0
	var scaledSendmats:Array[Mat] = null

	def start(learner0:Learner) = {
	  workerIP = InetAddress.getLocalHost;
	  learner = learner0;
	  if (learner != null) {
	    model = learner.model
		model.opts.trace = opts.trace
	    if (learner.modelmats == null) learner.init
	  }
	  executor = Executors.newFixedThreadPool(8);
	  listener = new CommandListener(opts.commandSocketNum, this);
	  listenerTask = executor.submit(listener);
	  intp = new ScriptEngineManager().getEngineByName("scala");
	  if (opts.trace > 0) logln(s"Worker started on ${workerIP.toString}")
	}

  def config(imach0:Int, gmods0:IMat, gridmachines0:IMat, workers0:Array[InetSocketAddress],
             masterSocketAddr0:InetSocketAddress, numModelMats0:Int, softmaxReduce:Boolean) = {
    val t1 = toc;
    imach = imach0;
    gmods = gmods0;
    gridmachines = gridmachines0;
    M = gridmachines.length;
    groups = new Groups(M, gmods.data, gridmachines.data, 0);
    masterSocketAddr = masterSocketAddr0;

    var machineOffset = 3
    val numMachines = numModelMats0 + (if (softmaxReduce) 1 else 0)
    if (machineArr == null) machineArr = new Array[Machine](numMachines)

    for (mmi <- 0 until numModelMats0) {
      if (machineArr(mmi) != null) machineArr(mmi).stop

      val machineBufsize = if (opts.bufsizes != null && mmi < opts.bufsizes.length) {
	opts.bufsizes(mmi)
      } else if (model != null && model.modelmats != null) {
	model.modelmats(mmi).length + model.modelmats(mmi).ncols
      } else {
	Worker.DEFAULT_BUFSIZE
      }

      val machine = startMachine(workers0, machineOffset, machineBufsize)
      machineArr(mmi) = machine
      machineOffset += 1
    }
    if (softmaxReduce) {
      val lastIdx = machineArr.length - 1
      if (machineArr(lastIdx) != null) machineArr(lastIdx).stop
      val machine = startMachine(workers0, machineOffset, M+1)
      machineArr(lastIdx) = machine
      opts.softmaxReduce = true
      lastSoftmax = ones(1, M) / M
    }

    this.synchronized { this.notify() } // config complete
    val t2 = toc
    if (opts.trace > 2) log("Machine config took %4.3f secs\n" format(t2-t1))
  }

  def startMachine(
    workerAddrs:Array[InetSocketAddress], machineOffset:Int, bufsize:Int
  ):Machine = {
    val workerMachineAddrs = workerAddrs.map(
      a => new InetSocketAddress(a.getAddress(), a.getPort() + machineOffset))

    val machine = new Machine(null, groups, imach, M, opts.useLong, bufsize, false,
      opts.machineTrace, opts.replicate, workerMachineAddrs);

    machine.configTimeout = opts.configTimeout;
    machine.reduceTimeout = opts.reduceTimeout;
    machine.sendTimeout = opts.sendTimeout;
    machine.recvTimeout = opts.recvTimeout;

    machine.sockBase = opts.peerSocketNum;
    machine.start(machine.maxk);

    machine
  }

  def permute(seed:Long) = {
    machineArr(0).groups.permute(seed.toInt);
  }

  def allReduceAccuracies(round:Int):FMat = {
    val accVec = zeros(1, M)

    val avgEndIdx = learner.reslist.length
    val avgStartIdx = Math.max(0, avgEndIdx - opts.accuracyAvgNBack)
    if (avgEndIdx > 0) {
      var score = Learner.scoreSummary(learner.reslist, avgStartIdx, avgEndIdx, 0)
      if (score.isNaN || score.isInfinity) {
        if (opts.trace > 0) logln(
	  s"Got NaN score! avgStartIdx: ${avgStartIdx}, avgEndIdx: ${avgEndIdx}")
        score = lastScore
      } else {
        lastScore = score
      }

      accVec(0, imach) = score
      val indexMat = irow(0 -> accVec.ncols)
      val result = machineAllreduce(machineArr.last, indexMat, accVec, opts.fuseConfigReduce)

      var smax = softmax(new FMat(1, M, result))
      val smaxSum = sum(lastSoftmax)(0)

      smax = if (smaxSum.isNaN || smaxSum.isInfinity) {
        if (opts.trace > 0) logln(s"Got NaN in softmax! result: ${result}, softmax: ${smax}")
        lastSoftmax
      } else {
        lastSoftmax = smax
        smax
      }
    } else {
      lastSoftmax = ones(1, M) / M
    }
    lastSoftmax
  }

  def allReduce(round:Int, limit:Long, tag:String):Unit = {
    val matIdxPat = """matIdx(\d+)""".r
    val matIdx = tag match {
       case matIdxPat(istr) => istr.toInt
       case _ => -1
    }

    if (matIdx == -1) {
      if (opts.trace > 0)
	logln("Worker got tagged allReduce response %s with invalid tag" format (tag))
    } else {
      allReduce(round, limit, matIdx)
    }
  }

  def allReduce(round:Int, limit:Long, matIdx:Int = 0) = {
    if (model != null && model.modelmats.asInstanceOf[AnyRef] != null) {
      val t1 = toc
      val doAvg = !opts.softmaxReduce && opts.doAvg
      model.snapshot(limit.toInt, doAvg, matIdx)
      val mm = model.modelmats(matIdx)
      val sendmatFull = model.sendmats(matIdx)

      val sendmat = if (!opts.softmaxReduce) {
	sendmatFull.view(sendmatFull.nrows, math.min(limit.toInt, sendmatFull.ncols))
      } else {
	if (scaledSendmats == null) scaledSendmats = new Array[Mat](model.modelmats.length)
	if (scaledSendmats(matIdx) == null) {
	  scaledSendmats(matIdx) = FMat(sendmatFull.nrows, sendmatFull.ncols)
	}
	val scaledSendmat = scaledSendmats(matIdx)
        scaledSendmat ~ sendmatFull * lastSoftmax(imach) // scale sendmat by softmax
      }

      val indexmat =
        if (model.indexmats != null && model.indexmats(matIdx).asInstanceOf[AnyRef] != null) {
          model.indexmats(matIdx)
        } else {
          irow(0 -> sendmat.ncols)
        }

      if (opts.trace > 3) logln("Starting allreduce for mm %d with dims %s datalen %d" format (
	matIdx, Arrays.toString(sendmat.dims.data), sendmat.length))
      val result = machineAllreduce(machineArr(matIdx), indexmat, sendmat, opts.fuseConfigReduce)

      if (model.recvmats == null) model.recvmats = new Array[FMat](model.modelmats.length)
      model.recvmats(matIdx) = new FMat(sendmat.nrows, mm.ncols, result)
      if (opts.doElastic) {
        model.elasticStep(limit.toInt, doAvg, opts.elasticAlpha, matIdx)
      } else {
        model.addStep(limit.toInt, doAvg, matIdx)
      }
      val t2 = toc
      val nbytes = indexmat match {
        case im:IMat => math.min(limit, im.length)*(2 + 2*sendmat.nrows)*8f
        case im:LMat => math.min(limit, im.length)*(4 + 2*sendmat.nrows)*8f
      }
      if (opts.trace > 2) log("Allreduce %5.2f MB took %5.4f secs at %5.2f MB/sec\n" format (
        nbytes/1e6f, t2-t1, nbytes/(t2-t1)/1e6f))
    } else {
      if (opts.trace > 2) log("Allreduce model is null\n")
    }
  }

  def machineAllreduce(
    machine:Machine, indexmat:Mat, sendmat:Mat, configReduce:Boolean
  ):Array[Float] = {
      return if (configReduce) {
        (indexmat, sendmat) match {
          case (lmat:LMat, fsendmat:FMat) => machine.configReduce(
	    lmat.data, lmat.data, fsendmat.data, fsendmat.length, fsendmat.nrows, round)
          case (imat:IMat, fsendmat:FMat) => machine.configReduce(
	    imat.data, imat.data, fsendmat.data, fsendmat.length, fsendmat.nrows, round)
        }
      } else {
        (indexmat, sendmat) match {
	  case (lmat:LMat, fsendmat:FMat) => machine.config(lmat.data, lmat.data, round)
          case (imat:IMat, fsendmat:FMat) => machine.config(imat.data, imat.data, round)
        }
	val fsendmat = sendmat.asInstanceOf[FMat]
        machine.reduce(fsendmat.data, fsendmat.length, fsendmat.nrows, round)
      }
  }

  def stop = {
    listener.stop = true
    listenerTask.cancel(true)
    if (machineArr != null) {
      for (m <- machineArr) if (m != null) m.stop
    }
  }

  def shutdown = {
    executor.shutdownNow();
    val tt= toc;
  }

  def handleCMD(cmd:Command) = {
    try {
      _handleCMD(cmd)
    } catch {
      case e:Exception => {
        val expresp = new WorkerExceptionResponse(cmd.round, cmd.dest, Command.printStackTrace(e))
	expresp.encode
	sendMaster(expresp)
	throw e
      }
    }
  }

  def _handleCMD(cmd:Command) = {
    if (cmd.magic != Command.magic) {
      if (opts.trace > 0) log("Machine %d got message with bad magic number %d\n" format (imach, cmd.magic));
    }  else if (cmd.dest != imach) {
      	if (opts.trace > 0) log("Machine %d got message with bad destination %d\n" format (imach, cmd.dest));
    } else {
    	cmd.ctype match {
    	case Command.configCtype => {
		val newcmd = new ConfigCommand(0, cmd.dest, null, null, null, null, -1, -1, -1, false, cmd.clen, cmd.bytes);
    		newcmd.decode;
    		if (opts.trace > 2) log("Received %s\n" format newcmd.toString);
		config(newcmd.dest, newcmd.gmods, newcmd.gridmachines,
		  Host.hostPortsToInet(newcmd.workerIPs, newcmd.workerPorts),
		  Host.hostPortToInet(newcmd.masterIP, newcmd.masterResPort),
		  newcmd.numModelMats,
		  newcmd.softmaxReduce);
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
    		val newcmd = new AllreduceCommand(0, cmd.dest, 0, cmd.bytes, cmd.tag);
    		newcmd.decode;
    		if (opts.trace > 2) log("Received %s\n" format newcmd.toString);
		try {
		  allReduce(newcmd.round, newcmd.limit, newcmd.tag)
		  sendMaster(AllreduceResponse.success(cmd.round, cmd.dest, cmd.tag))
		} catch {
		  case e:Exception => {
		    if (opts.trace > 0) log("Allreduce failed:\n%s" format Command.printStackTrace(e));
		    sendMaster(AllreduceResponse.failure(cmd.round, cmd.dest, cmd.tag))
		  }
		}
    	}
    	case Command.permuteAllreduceCtype => {
    		val newcmd = new PermuteAllreduceCommand(0, cmd.dest, 0, 0, cmd.bytes, cmd.tag);
    		newcmd.decode;
    		if (opts.trace > 2) log("Received %s\n" format newcmd.toString);
    		permute(newcmd.seed);
    		allReduce(newcmd.round, newcmd.limit);
    		if (opts.respond > 0) sendMaster(new Response(Command.permuteAllreduceCtype, newcmd.round, imach));
    	}
    	case Command.accuracyAllreduceCtype => {
    		val newcmd = new AccuracyAllreduceCommand(0, cmd.dest, cmd.bytes)
    		newcmd.decode
    		if (opts.trace > 2) log("Received %s\n" format newcmd.toString)
		allReduceAccuracies(newcmd.round)
    		if (opts.respond > 0) sendMaster(new Response(Command.accuracyAllreduceCtype, newcmd.round, imach))
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
		  if (learner.modelmats == null) learner.init
    		  sendMaster(new Response(Command.startLearnerCtype, newcmd.round, imach))
		  if (opts.trace > 0) logln("Worker starting training.")
		  learner.train(false)
		  if (opts.trace > 0) logln("Worker training finished! Responding to Master.")
    		  sendMaster(new Response(Command.learnerDoneCtype, -1, imach))
    		} else {
		  logln("Recieved startLearner Command but learner == null")
		}
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
    		obj = newcmd.func(this);
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
			if (opts.trace > 3) log("Started CommandListener on %s:%d\n" format (localIP, socketnum));
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

  def ackReady = {
    // explicitly run on main thread
    val rw = new ResponseWriter(masterSocketAddr, new AckReadyResponse(imach), this);
    rw.run()
    if (opts.trace > 2) log("Send ackReady to %s\n" format(masterSocketAddr))
  }

  def registerWorker(masterSocketAddr:InetSocketAddress) = {
    // explicitly run on main thread
    val registerResp = new RegisterWorkerResponse(opts.commandSocketNum)
    registerResp.encode
    val rw = new ResponseWriter(masterSocketAddr, registerResp, this);
    rw.run()
    if (opts.trace > 1) {
      log("Sent registration to %s\n" format(masterSocketAddr))
      log("Waiting for config on %s...\n" format(
	new InetSocketAddress(localIP, opts.commandSocketNum)))
    }
    this.synchronized { this.wait() }
    log("Recieved config! Continuing.\n")
  }

  def signalLearnerDone = {
    // explicitly run on main thread
    val doneResp = new Response(Command.learnerDoneCtype, -1, imach)
    doneResp.encode
    val rw = new ResponseWriter(masterSocketAddr, doneResp, this);
    rw.run()
    if (opts.trace > 1) log("Signaled LearnerDone")
  }
}

object Worker {
  val DEFAULT_BUFSIZE = 10*1000000

  trait Opts extends Host.Opts {
    var machineSocketNum = peerSocketNum + 1;
    var configTimeout = 3000;
    var reduceTimeout = 3000;
    var cmdTimeout = 1000;
    var fuseConfigReduce = false;
    var doAvg = true
    var doElastic = false
    var elasticAlpha = 0.9 / 4
    var useLong = false
    var replicate = 1;
    var bufsizes:Array[Int] = null;
    var respond = 0;
    var softmaxReduce = false
    var accuracyAvgNBack = 5
  }

  class Options extends Opts {}
}
