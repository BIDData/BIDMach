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
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;


class Command(val ctype:Int, val clen:Int, val bytes:Array[Byte]) {
  val magic = Command.magic;
  val byteData = ByteBuffer.wrap(bytes);
  val intData = byteData.asIntBuffer;
  val floatData = byteData.asFloatBuffer;
  val longData = byteData.asLongBuffer;
  var imach = 0;
  
  def encode()
  def decode()
  
}

object Command {
	val magic = 0xa6b38734;
	val configCtype = 1;
	val permuteCtype = 2;
	val allreduceCtype = 3;
}

class CommandWriter(val me:Master, dest:String, socketnum:Int, command:Command) extends Runnable {

	def run() {
		var socket:Socket = null;
		//			log(String.format("M %d W %d Running writer %s\n", imachine, dest, this.toString()));
		try {
			socket = new Socket();
			socket.connect(new InetSocketAddress(dest, socketnum), me.sendTimeout);
			if (socket.isConnected()) {
				val ostr = new DataOutputStream(socket.getOutputStream());
				ostr.writeInt(command.magic)
				ostr.writeInt(command.ctype);
				ostr.writeInt(command.clen);
				ostr.write(command.bytes, 0, command.clen*4);		
			}
		}	catch {
		case e:Exception =>
		if (me.trace > 0) me.log("Master problem sending command %s\n" format command.toString);
		} finally {
			try { if (socket != null) socket.close(); } catch {
		case e:Exception =>
		if (me.trace > 0) me.log("Master problem closing socket\n");			  
		}
		}
	}
}

class CommandReader(val me:Worker, socket:Socket) extends Runnable {
	def run() {
		try {
			val istr = new DataInputStream(socket.getInputStream());
			val magic = istr.readInt();
			val ctype = istr.readInt();
			val clen = istr.readInt();
			val cmd = new Command(ctype, clen, new Array[Byte](clen*4));
			if (me.trace > 2) me.log("Worker %d got packet %s\n" format (me.imach, cmd.toString));
			var waiting = 0;
			while (waiting < me.cmdTimeout) {
				Thread.sleep(10);
				waiting += 10;
			}
			istr.readFully(cmd.bytes, 0, clen*4);
			me.handleCMD(cmd);
		} catch {
		  case e:Exception =>	if (me.trace > 0) me.log("Worker %d Problem reading socket "+e.toString()+"\n" format (me.imach));
		} finally {
			try {
			  socket.close();
			} catch {
			  case e:IOException => {}
			}
		}
  }
}

class CommandListener(val me:Worker, val socketnum:Int) extends Runnable {
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
				val scs = new CommandReader(me, ss.accept());
				val fut = me.machExecutor.submit(scs);
			} catch {
			case e:SocketException => {
			}
			// This is probably due to the server shutting to. Don't do anything.
			case e:Exception => {
				if (me.trace > 0) me.log("Machine %d Command listener had a problem "+e format me.imach);
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
				if (me.trace > 0) me.log("Machine %d trouble closing command listener" format me.imach);
			}
			}			
		}
	}
}

class ConfigCommand(clen:Int) extends Command(Command.configCtype, clen, new Array[Byte](clen*4)) {
  
  var gmods:IMat = null;
  var gridmachines:IMat = null;
  var workerIPs:IMat = null;
  
  def setFields(imach0:Int, gmods0:IMat, gridmachines0:IMat, workerIPs0:IMat) {
    imach = imach0;
    gmods = gmods;
    gridmachines = gridmachines0;
    workerIPs = workerIPs0;
  }
  
  override def encode ():Unit = {
  	intData.rewind();
  	intData.put(imach);
  	intData.put(gmods.length);
  	intData.put(gmods.data, 0, gmods.length);
  	intData.put(gridmachines.length);
  	intData.put(gridmachines.data, 0, gridmachines.length);
  	intData.put(workerIPs.length);
  	intData.put(workerIPs.data, 0, workerIPs.length);
  }
  
  override def decode():Unit = {
  	intData.rewind();
    imach = intData.get();
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
}

class PermuteCommand() extends Command(Command.permuteCtype, 2, new Array[Byte](2*4)) {
  
  var seed:Long = 0;
  
  def setFields(seed0:Long) {
    seed = seed0;
  }
  
  override def encode ():Unit = {
  	longData.rewind();
  	longData.put(seed);
  }
  
  override def decode():Unit = {
  	longData.rewind();
    seed = longData.get();    
  }
}

class AllreduceCommand() extends Command(Command.allreduceCtype, 4, new Array[Byte](4*4)) {
  
  var round:Int = 0;
  var limit:Long = 0;
  
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
}

class PermuteAllreduceCommand() extends Command(Command.allreduceCtype, 6, new Array[Byte](6*4)) {
  
  var seed:Long = 0;
  var round:Int = 0;
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
}




