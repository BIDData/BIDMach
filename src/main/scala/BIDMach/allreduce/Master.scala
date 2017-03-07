package BIDMach.allreduce

import BIDMat.{Mat,SBMat,CMat,DMat,FMat,IMat,HMat,GDMat,GLMat,GMat,GIMat,GSDMat,GSMat,LMat,SMat,SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import edu.berkeley.bid.comm._
import scala.collection.parallel._
import scala.util.control.Breaks._
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Callable;
import java.util.concurrent.Future;
import java.net.ServerSocket;
import java.net.Socket;
import java.net.SocketException;
import java.net.BindException;
import java.net.InetAddress;
import java.net.InetSocketAddress;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import javax.script.ScriptEngine;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.{ConcurrentHashMap, Semaphore}
import BIDMach.models.Model
import scala.collection.mutable.{Map => MutableMap, HashMap => MutableHashMap}
import java.io.ByteArrayOutputStream
import java.io.PrintStream


class Master(override val opts:Master.Opts = new Master.Options) extends Host {
  
  var listener:ResponseListener = null;
	var listenerTask:Future[_] = null;
	var reducer:Reducer = null
	var reduceTask:Future[_] = null
	var numLearnersStarted:Integer = 0
	var numLearnersFinished:Integer = 0
	var numAckedReady:Integer = 0
	var activeCommand:Command = null;
	var activeTaggedCommands:ConcurrentHashMap[String, Command] =
	  new ConcurrentHashMap[String, Command]()
	var responses:IMat = null;
	var results:Array[AnyRef] = null;
        var masterIP:InetAddress = null;
	var nresults = 0;
        var allreduceTimer:Long = 0;
	var numExpectedRegisteredWorkers:Int = -1
	var numRegisteredWorkers:AtomicInteger = new AtomicInteger()
	var allWorkersRegisteredSema:Semaphore = new Semaphore(0)
	
	def init() {
    masterIP = InetAddress.getLocalHost;
	  executor = Executors.newFixedThreadPool(opts.numThreads);
	  listener = new ResponseListener(opts.responseSocketNum, this);
	  listenerTask = executor.submit(listener);
	}
	
	def readConfig(configDir:String) {
		val clengths = loadIMat(configDir + "dims.imat.lz4");
		val allgmods = loadIMat(configDir + "gmods.imat.lz4");
		val allmachinecodes = loadIMat(configDir + "machines.imat.lz4");
		gmods = allgmods(0->clengths(M-1), M-1);
		gridmachines = allmachinecodes(0->M, M-1);
	}
  
  def config(gmods0:IMat, gridmachines0:IMat, workers0:Array[InetSocketAddress]) {
    gmods = gmods0;
    gridmachines = gridmachines0;
    workers = workers0;
    M = workers.length;
    responses = izeros(1,M+1);
    results = new Array[AnyRef](M);
    nresults = 0;
  }
  
  def sendConfig() {
    val cmd = new ConfigCommand(
      round, 0, gmods, gridmachines, workers, masterIP, opts.responseSocketNum)
    broadcastCommand(cmd);
  }

  def startUpdatesAfterRegistration(numExpectedWorkers:Int) {
    numExpectedRegisteredWorkers = numExpectedWorkers
    config(
      irow(numExpectedWorkers),
      irow(0->numExpectedWorkers),
      new Array[InetSocketAddress](numExpectedWorkers)
    )
    executor.submit(new Registrar())
  }
  
  def permuteNodes(seed:Long) {
    val cmd = new PermuteCommand(round, 0, seed);
    broadcastCommand(cmd);
  }
  
  def startUpdates(waitForAck:Boolean = false) {
    reducer = new Reducer(waitForAck)
    reduceTask = executor.submit(reducer)
    if (opts.trace > 2) log("Machine threshold: %5.2f\n" format opts.machineThreshold*M)
  }

  def startUpdates:Unit = startUpdates(false)
  
  def stopUpdates() {
    if (reducer == null || reducer.stop) {
      if (opts.trace > 2) logln("Reducer was not running\n")
    } else {
      reducer.stop = true
      reduceTask.cancel(true)
      logln("Stopped allreduce!")
    }
    numLearnersFinished.synchronized {
      numLearnersFinished += 1
      if (numLearnersFinished == M)
      logln("Total distributed time: %.3fs" format (
	(System.currentTimeMillis - allreduceTimer) / 1000.0))
    }
  }
  
  def startLearners(blocking:Boolean = false) {
    val cmd = new StartLearnerCommand(round, 0);
    broadcastCommand(cmd);
    allreduceTimer = System.currentTimeMillis;
    if (blocking) {
      this.synchronized { this.wait() }
    }
  }

  def startLearners:Unit = startLearners(false)
  
  def permuteAllreduce(round:Int, limit:Int) {
  	val cmd = new PermuteAllreduceCommand(round, 0, round, limit, null);
  	broadcastCommand(cmd);
  }
  
  def parAssign(obj:AnyRef, str:String, timesecs:Int = 10):Array[AnyRef] = {
    val cmd = new AssignObjectCommand(round, 0, obj, str);
    for (i <- 0 until M) results(i) = null;
    nresults = 0;
    broadcastCommand(cmd);
    var tmsec = 0;
    while (nresults < M && tmsec < timesecs * 1000) {
      Thread.`sleep`(10);
      tmsec += 10;
    }
    results.clone
  }

  def parEval(str:String, timesecs:Int = 10):Array[AnyRef] = {
    val cmd = new EvalStringCommand(round, 0, str);
    for (i <- 0 until M) results(i) = null;
    nresults = 0;
    broadcastCommand(cmd); 
    var tmsec = 0;
    while (nresults < M && tmsec < timesecs * 1000) {
      Thread.`sleep`(10);
      tmsec += 10;
    }
    results.clone
  }
  
  def parCall(func:(Worker) => AnyRef, timesecs:Int = 10):Array[AnyRef] = {
    val cmd = new CallCommand(round, 0, func);
    for (i <- 0 until M) results(i) = null;
    nresults = 0;
    broadcastCommand(cmd); 
    var tmsec = 0;
    while (nresults < M && tmsec < timesecs * 1000) {
      Thread.`sleep`(10);
      tmsec += 10;
    }
    results.clone
  }
  
  def setMachineNumbers {
  	if (opts.trace > 2) log("Broadcasting setMachineNumbers\n");
  	val futures = new Array[Future[_]](M);
  	val timeout = executor.submit(new TimeoutThread(opts.sendTimeout, futures));
  	for (imach <- 0 until M) {
  	  val cmd = new SetMachineCommand(round, 0, imach);
  	  cmd.encode
  		futures(imach) = send(cmd, workers(imach));   
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
  	timeout.cancel(true);
  }
    
  def broadcastCommand(cmd:Command) {
    cmd.encode
    if (cmd.tag != null) {
      activeTaggedCommands.put(cmd.tag, cmd)
    } else {
      activeCommand = cmd
    }
    if (opts.trace > 2) log("Broadcasting cmd %s\n" format cmd)

    val futures = new Array[Future[_]](M)
    responses.clear
    val timeout = executor.submit(new TimeoutThread(opts.sendTimeout, futures))
    for (imach <- 0 until M) {
      val newcmd = new Command(cmd.ctype, round, imach, cmd.clen, cmd.bytes, cmd.blen, cmd.tag)
      futures(imach) = send(newcmd, workers(imach))
    }
    for (imach <- 0 until M) {
      try {
        futures(imach).get()
      } catch {
        case e:Exception => {}
      }
      if (futures(imach).isCancelled()) {
        if (opts.trace > 0) log("Broadcast to machine %d timed out, cmd %s\n" format (imach, cmd))
      }
    }
    timeout.cancel(true)
  }
  
  def almostDone(threshold:Float = 0.75f):Boolean = {
    val c = responses.synchronized { responses(M); }
    c >= M * threshold;
  }
  
  def send(cmd:Command, address:InetSocketAddress):Future[_] = {
    val cw = new CommandWriter(address, cmd, this);
    executor.submit(cw);
  }

  class Registrar() extends Runnable {
    def run() {
      if (opts.trace > 1) log("Waiting for %d workers to register...\n" format M)
      allWorkersRegisteredSema.acquire() // block until all registrations are recieved
      if (opts.trace > 1) log("%d workers successfully registered! Starting updates.\n" format M)
      sendConfig
      startUpdates()
    }
  }

  class Reducer(waitForAck:Boolean = false) extends Runnable {
    var stop = false
    var allreduceCollected:Int = 0
    var allreduceStartTime:Long = 0

    def run() {
      try {

      if (waitForAck) {
	if (opts.trace > 1) log("Waiting for %d ready acks...\n" format M)
	numAckedReady.synchronized {
	  if (numAckedReady < M) numAckedReady.wait()
	}
	if (opts.trace > 1) log("Recieved all ready acks! Starting Reducer.\n" )
      }
      var limit = 0
      while (!stop) {
	val newlimit0 = if (opts.limitFctn != null) {
	  opts.limitFctn(round, opts.limit)
	} else {
	  opts.limit
	}
	limit = if (newlimit0 <= 0) 2000000000 else newlimit0
	val cmd = if (opts.permuteAlways) {
	  new PermuteAllreduceCommand(round, 0, round, limit, "foo")
	} else {
	  new AllreduceCommand(round, 0, limit, "foo")
	}
	this.synchronized {
	  allreduceCollected = 0
	}
	broadcastCommand(cmd)

	allreduceStartTime = System.currentTimeMillis
	this.synchronized {
	  this.wait(opts.timeThresholdMsec)
	}
	if (opts.trace > 1) log("Finished round %d, workers %d/%d, time %dms\n" format(
	  round, allreduceCollected, M, System.currentTimeMillis - allreduceStartTime))
	round += 1

      }
    } catch {
      case e:InterruptedException => {
	if (!stop) logln("Reducer failed: %s" format Host.printStackTrace(e))
      }
      case e:Exception => logln("Reducer failed: %s" format Host.printStackTrace(e))
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
  
  def inctable(mat:IMat, src:Int) = {
    mat.synchronized {
    	mat(0, src) += 1;
    	mat(0, M) += 1;
    }
  }
  
  def addObj(obj:AnyRef, src:Int) = {
    results.synchronized {
    	results(src) = obj;
    	nresults += 1;
    }
  }
  
  def handleResponse(resp:Response) = {
    if (resp.magic != Response.magic) {
      if (opts.trace > 0) log("Master got response with bad magic number %d\n" format (resp.magic));

    } else if (resp.rtype == Command.learnerDoneCtype) {
      stopUpdates()

    } else if (resp.tag != null && activeTaggedCommands.get(resp.tag) != null) {
      val activeTaggedCmd = activeTaggedCommands.get(resp.tag)
      if ((resp.rtype == Command.allreduceCtype || resp.rtype == Command.permuteAllreduceCtype)
	  && resp.round == activeTaggedCmd.round) {
	handleAllReduceResponse(resp)

      } else if (opts.trace > 0) {
	log("Master got tagged response %s with bad type/round (%d,%d), should be (%d,%d)\n"
	    format (resp.tag, resp.rtype, resp.round, activeCommand.ctype, activeCommand.round))
      }

    } else if (activeCommand != null) {
      if (resp.rtype == activeCommand.ctype && resp.round == activeCommand.round) {
    	inctable(responses, resp.src);
	resp.rtype match {
	  case Command.startLearnerCtype => {
	    this.synchronized {
              numLearnersStarted += 1
	      if (numLearnersStarted >= M) this.notify()
	    }
	  }
	}

      } else if ((activeCommand.ctype == Command.evalStringCtype || activeCommand.ctype == Command.callCtype)
	         && resp.rtype == Command.returnObjectCtype && resp.round == activeCommand.round) {
        val newresp = new ReturnObjectResponse(resp.round, resp.src, null, resp.bytes);
	newresp.decode;
	addObj(newresp.obj, resp.src)
	if (opts.trace > 2) log("Received %s\n" format newresp.toString);
      } else if (opts.trace > 0) {
	log("Master got response with bad type/round (%d,%d), should be (%d,%d)\n"
	    format (resp.rtype, resp.round, activeCommand.ctype, activeCommand.round))
      }

    } else {
      // ackReady and registerWorker fall here
      if (resp.rtype == Command.registerWorkerCtype) {
	val registerResp = new RegisterWorkerResponse(resp.bytes)
	registerResp.decode
	val workerNum = numRegisteredWorkers.getAndIncrement()
	workers(workerNum) = registerResp.workerCmdSocketAddr

	// set worker number on this thread
	if (opts.trace > 0) {
	  log("Recieved registration from worker %d @ %s\n" format
	    (workerNum, registerResp.workerCmdSocketAddr))
	}
        val setMachineCmd = new SetMachineCommand(round, 0, workerNum)
	setMachineCmd.encode
	(new CommandWriter(
	  registerResp.workerCmdSocketAddr,
	  setMachineCmd,
	  this)).run()

	if (workerNum == numExpectedRegisteredWorkers - 1) {
	  // all expected workers registered!
	  allWorkersRegisteredSema.release()
	}
      } else if (resp.rtype == Command.ackReadyCtype) {
	numAckedReady.synchronized {
	  numAckedReady += 1
	  if (numAckedReady >= M) {
	    numAckedReady.notifyAll()
	  }
	}
      }
    }
  }

  def handleAllReduceResponse(resp:Response) {
    val matIdxPat = """matIdx(\d+)""".r
    // val matIdx = resp.tag match {
    //    case matIdxPat(matIdx) => matIdx.toInt
    //    case _ => -1
    // }
    val matIdx = 0

    if (matIdx == -1) {
      if (opts.trace > 0)
	logln("Master got tagged allReduce response %s with invalid tag" format (resp.tag))
    } else {
      reducer.synchronized {
        reducer.allreduceCollected += 1
	val delta = System.currentTimeMillis - reducer.allreduceStartTime
	if (delta < opts.minWaitTime) {
	  if (reducer.allreduceCollected >= M) reducer.notify()
	} else if (delta < opts.timeThresholdMsec) {
	  if (reducer.allreduceCollected >= M*opts.machineThreshold) reducer.notify()
	}
      }
    }
  }

class ResponseListener(val socketnum:Int, me:Master) extends Runnable {
  var stop = false;
  var ss:ServerSocket = null;

  def start() {
    try {
      ss = new ServerSocket(socketnum);
    } catch {
      case e:BindException => throw e
      case e:Exception => {
        if (opts.trace > 0)
	  log("Problem in ResponseListener\n%s" format Response.printStackTrace(e))
      }
    }
  }

  def run() {
    start();
    while (!stop) {
      try {
	val scs = new ResponseReader(ss.accept(), me);
	if (opts.trace > 2) log("Command Listener got a message\n");
	val fut = executor.submit(scs);
      } catch {
        case e:SocketException => {
          if (opts.trace > 0) log("Problem starting a socket reader\n%s" format Response.printStackTrace(e));
        }
        // This is probably due to the server shutting to. Don't do anything.
        case e:Exception => {
          if (opts.trace > 0) {
            log("Master Response listener had a problem\n%s" format Response.printStackTrace(e));
            Thread.`sleep`(1000)
          }
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
}

object Master {
	trait Opts extends Host.Opts{
		var limit = 0;
		var limitFctn:(Int,Int)=>Int = null;
		var intervalMsec = 1000;
		var timeScaleMsec = 1e-4f;
		var permuteAlways = true;
		var numThreads = 16;
    var machineThreshold = 0.75;
    var minWaitTime = 3000;
    var timeThresholdMsec = 5000;
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

