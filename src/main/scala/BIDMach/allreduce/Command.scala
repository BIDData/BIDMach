package BIDMach.allreduce

import BIDMat.{Mat,SBMat,CMat,DMat,FMat,IMat,HMat,GDMat,GLMat,GMat,GIMat,GSDMat,GSMat,LMat,SMat,SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import edu.berkeley.bid.comm._
import scala.collection.parallel._
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.net.ServerSocket;
import java.net.Socket;
import java.net.InetSocketAddress;
import java.net.SocketException;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.PrintStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;


class Command(val ctype:Int, round0:Int, dest0:Int, val clen:Int, val bytes:Array[Byte]) {
  val magic = Command.magic;
  var dest = dest0;
  var round = round0;
  val byteData = ByteBuffer.wrap(bytes);
  val intData = byteData.asIntBuffer;
  val floatData = byteData.asFloatBuffer;
  val longData = byteData.asLongBuffer;
  
  def encode() = {}
  def decode() = {}
  
  def this(ctype0:Int, round0:Int, dest0:Int, clen0:Int) = this(ctype0, round0, dest0, clen0, new Array[Byte](4*clen0));
  
  override def toString():String = {
    "Command %s, length %d bytes" format (Command.names(ctype), clen*4);
  }
  
}

class CommandWriter(dest:String, socketnum:Int, command:Command, me:Master) extends Runnable {

	def run() {
		var socket:Socket = null;
	  try {
	  	socket = new Socket();
	  	socket.setReuseAddress(true);
	  	socket.connect(new InetSocketAddress(dest, socketnum), me.opts.sendTimeout);
	  	if (socket.isConnected()) {
	  		val ostr = new DataOutputStream(socket.getOutputStream());
	  		ostr.writeInt(command.magic);
	  		ostr.writeInt(command.ctype);
	  		ostr.writeInt(command.round);
	  		ostr.writeInt(command.dest);
	  		ostr.writeInt(command.clen);
	  		ostr.write(command.bytes, 0, command.clen*4);		
	  	}
	  }	catch {
	  case e:Exception =>
	  if (me.opts.trace > 0) {
	  	me.log("Master problem sending command %s\n%s\n" format (command.toString, Command.printStackTrace(e)));
	  }
	  } finally {
	  	try { if (socket != null) socket.close(); } catch {
	  	case e:Exception =>
	  	if (me.opts.trace > 0) me.log("Master problem closing socket\n%s\n" format Command.printStackTrace(e));			  
	  	}
	  }
	}
}

class CommandReader(socket:Socket, me:Worker) extends Runnable {
	def run() {
		try {
			val istr = new DataInputStream(socket.getInputStream());
			val magic = istr.readInt();
			val ctype = istr.readInt();
			val round = istr.readInt();
			val dest = istr.readInt();
			val clen = istr.readInt();
			val cmd = new Command(ctype, round, dest, clen, new Array[Byte](clen*4));
			if (me.opts.trace > 2) me.log("Worker %d got packet %s\n" format (me.imach, cmd.toString));
			istr.readFully(cmd.bytes, 0, clen*4);
			try {
				socket.close();
			} catch {
			case e:IOException => {if (me.opts.trace > 0) me.log("Worker %d Problem closing socket "+Command.printStackTrace(e)+"\n" format (me.imach))}
			}
			me.handleCMD(cmd);
		} catch {
		case e:Exception =>	if (me.opts.trace > 0) me.log("Worker %d Problem reading socket "+Command.printStackTrace(e)+"\n" format (me.imach));
		} finally {
			try {
				if (!socket.isClosed) socket.close();
			} catch {
			case e:IOException => {if (me.opts.trace > 0) me.log("Worker %d Final Problem closing socket "+Command.printStackTrace(e)+"\n" format (me.imach))}
			}
		}
	}
}

object Command {
	val magic = 0xa6b38734;
	final val configCtype = 1;
	final val permuteCtype = 2;
	final val allreduceCtype = 3;
	final val permuteAllreduceCtype = 4;
	final val setMachineCtype = 5;
	final val startLearnerCtype = 6;
	final val learnerDoneCtype = 7;
	final val sendObjectCtype = 8;
	final val sendLearnerCtype = 8;
	final val names = Array[String]("", "config", "permute", "allreduce", "permuteAllreduce", "setMachine", "startLearner", "learnerDone", "sendObject", "sendLearner");
	
	  
  def toAddress(v:Int):String = {
    val p0 = (v >> 24) & 255;
    val p1 = (v >> 16) & 255;
    val p2 = (v >> 8) & 255;
    val p3 = v & 255;
    "%d.%d.%d.%d" format(p0,p1,p2,p3);
   }
  
  def address(a:Int, b:Int, c:Int, d:Int):Int = {
    d + ((c + ((b + (a << 8)) << 8)) << 8);
  }
  
  def printStackTrace(e:Exception):String = {
    val baos = new ByteArrayOutputStream();
    val ps = new PrintStream(baos);
    e.printStackTrace(ps);
    val str = baos.toString();
    ps.close();
    str;
  }
}

class ConfigCommand(round0:Int, dest0:Int, clen:Int, bytes:Array[Byte]) extends Command(Command.configCtype, round0, dest0, clen, bytes) {
  
  var gmods:IMat = null;
  var gridmachines:IMat = null;
  var workerIPs:IMat = null;
  
  def this(round0:Int, dest0:Int, clen0:Int) = this(round0, dest0, clen0, new Array[Byte](clen0*4));
  
  def setFields(imach0:Int, gmods0:IMat, gridmachines0:IMat, workerIPs0:IMat) {
    dest = imach0;
    gmods = gmods;
    gridmachines = gridmachines0;
    workerIPs = workerIPs0;
  }
  
  override def encode ():Unit = {
  	intData.rewind();
  	intData.put(gmods.length);
  	intData.put(gmods.data, 0, gmods.length);
  	intData.put(gridmachines.length);
  	intData.put(gridmachines.data, 0, gridmachines.length);
  	intData.put(workerIPs.length);
  	intData.put(workerIPs.data, 0, workerIPs.length);
  }
  
  override def decode():Unit = {
  	intData.rewind();
    val lgmods = intData.get();
    gmods = izeros(lgmods,1);
    intData.get(gmods.data, 0, lgmods);
    val lgm = intData.get();
    gridmachines = izeros(lgm, 1);
    intData.get(gridmachines.data, 0, lgm);
    val lwips = intData.get();
    workerIPs = izeros(lwips, 1);
    intData.get(workerIPs.data, 0, lwips);    
  }
  
  override def toString():String = {
    var ostring = new StringBuilder("Command %s, length %d words" format (Command.names(ctype), clen));
    ostring.append("\nGroups: ")
    for (i <- 0 until gmods.length) {
      ostring.append("%d " format gmods(i));
    }
    ostring.append("\nGridmachines: ");
    for (i <- 0 until math.min(20, gridmachines.length)) {
      ostring.append("%d " format gridmachines(i));
    }
    ostring.append("\nWorkerIPs: ");
    for (i <- 0 until math.min(20, gridmachines.length)) {
      ostring.append("%s " format Command.toAddress(workerIPs(i)));
    }
    ostring.append("\n")
    ostring.toString;
  }
}

class PermuteCommand(round0:Int, dest0:Int, bytes:Array[Byte]) extends Command(Command.permuteCtype, round0, dest0, 2, bytes) {
  
  var seed:Long = 0;
  
  def this(round0:Int, dest0:Int) = this(round0, dest0, new Array[Byte](2*4));
  
  def setFields(round0:Int, seed0:Long) {
    round = round0;
    seed = seed0;
  }
  
  override def encode ():Unit = {
  	longData.rewind();
  	longData.put(round);
  	longData.put(seed);
  }
  
  override def decode():Unit = {
  	longData.rewind();
  	round = longData.get().toInt;
    seed = longData.get();    
  }
  
  override def toString():String = {
     "Command %s, length %d words, seed %d" format (Command.names(ctype), clen, seed);
  }
}

class SetMachineCommand(round0:Int, dest0:Int, newdest0:Int, bytes:Array[Byte]) extends Command(Command.setMachineCtype, round0, dest0, 1, bytes) {
  
  dest = dest0;
  var newdest = newdest0;
  
  def this(round0:Int, dest0:Int, newdest0:Int) = this(round0, dest0, newdest0, new Array[Byte](1*4));
  
  override def encode ():Unit = {
  	intData.rewind();
  	intData.put(newdest);
  }
  
  override def decode():Unit = {
  	intData.rewind();
    newdest = intData.get();    
  }
  
  override def toString():String = {
     "Command %s, length %d words, machine %d newdest %d" format (Command.names(ctype), clen, dest, newdest);
  }
}

class StartLearnerCommand(round0:Int, dest0:Int, bytes:Array[Byte]) extends Command(Command.startLearnerCtype, round0, dest0, 1, bytes) {
  
  dest = dest0;
  
  def this(round0:Int, dest0:Int) = this(round0, dest0, new Array[Byte](1*4));
  
  override def encode ():Unit = {
  	intData.rewind();
  	intData.put(dest);
  }
  
  override def decode():Unit = {    
  }
  
  override def toString():String = {
     "Command %s, length %d words, machine %d" format (Command.names(ctype), clen, dest);
  }
}

class AllreduceCommand(round0:Int, dest0:Int, bytes:Array[Byte]) extends Command(Command.allreduceCtype, round0, dest0, 4, bytes) {
  
  var limit:Long = 0;
  
  def this(round0:Int, dest0:Int) = this(round0, dest0, new Array[Byte](4*4));
  
  def setFields(round0:Int, limit0:Long) {
    round = round0;
    limit = limit0;
  }
  
  override def encode():Unit = {
  	longData.rewind();
  	longData.put(round);
  	longData.put(limit);
  }
  
  override def decode():Unit = {
  	longData.rewind();
  	round = longData.get().toInt;
    limit = longData.get();    
  }
  
  override def toString():String = {
     "Command %s, length %d words, round %d limit %d" format (Command.names(ctype), clen, round, limit);
  }
}

class PermuteAllreduceCommand(round0:Int, dest0:Int, bytes:Array[Byte]) extends Command(Command.permuteAllreduceCtype, round0, dest0, 6, bytes) {
  
	def this(round0:Int, dest0:Int) = this(round0, dest0, new Array[Byte](6*4));
  
  var seed:Long = 0;
  var limit:Long = 0;
  
  def setFields(round0:Int, seed0:Long, limit0:Long) {
    round = round0;
    seed = seed0;
    limit = limit0;
  }
  
  override def encode():Unit = {
  	longData.rewind();
  	longData.put(round);
  	longData.put(seed);
  	longData.put(limit);
  }
  
  override def decode():Unit = {
  	longData.rewind();
  	round = longData.get().toInt;
  	seed = longData.get();
    limit = longData.get();    
  }
  
  override def toString():String = {
     "Command %s, length %d words, round %d seed %d limit %d" format (Command.names(ctype), clen, round, seed, limit);
  }
}

class SendObjectCommand(round0:Int, dest0:Int, bytes:Array[Byte]) extends Command(Command.sendObjectCtype, round0, dest0, bytes.size, bytes) {
  
  dest = dest0;
  var obj:AnyRef = null;
  
  def this(round0:Int, dest0:Int, obj0:AnyRef) = {
    this(round0, dest0,  {
    	val out  = new ByteArrayOutputStream();
    	val output = new ObjectOutputStream(out);
    	output.writeObject(obj0);
    	output.close;
    	out.toByteArray()});
  }
  
  override def encode ():Unit = {
  }
  
  override def decode():Unit = {    
		val in = new ByteArrayInputStream(bytes);
		val input = new ObjectInputStream(in);
		obj = input.readObject;
		input.close;
  }
  
  override def toString():String = {
     "Command %s, length %d words, machine %d" format (Command.names(ctype), clen, dest);
  }
}

class SendLearnerCommand(round0:Int, dest0:Int, bytes:Array[Byte]) extends SendObjectCommand(round0, dest0, bytes) {
  override val ctype = Command.sendLearnerCtype;
}




