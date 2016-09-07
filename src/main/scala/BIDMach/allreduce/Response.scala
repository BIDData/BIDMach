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
import java.io.ByteArrayOutputStream;
import java.io.PrintStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;


class Response(val rtype:Int, val src:Int, val clen:Int, val bytes:Array[Byte]) {
  val magic = Response.magic;
  var round = 0;
  val byteData = ByteBuffer.wrap(bytes);
  val intData = byteData.asIntBuffer;
  val floatData = byteData.asFloatBuffer;
  val longData = byteData.asLongBuffer;
  
  def encode() = {}
  def decode() = {}
  
  def this(Rtype0:Int, dest0:Int, clen0:Int) = this(Rtype0, dest0, clen0, new Array[Byte](4*clen0));
  
  override def toString():String = {
    "Response %s, length %d bytes" format (Response.names(rtype), clen*4);
  }
  
}

object Response {
	val magic = 0xa6b38734;
	final val configRtype = 1;
	final val permuteRtype = 2;
	final val allreduceRtype = 3;
	final val permuteAllreduceRtype = 4;
	final val setMachineRtype = 5;
	final val learnerDoneRtype = 6;
	final val names = Array[String]("", "config", "permute", "allreduce", "permuteAllreduce", "setMachine", "learnerDone");
  
  def printStackTrace(e:Exception):String = {
    val baos = new ByteArrayOutputStream();
    val ps = new PrintStream(baos);
    e.printStackTrace(ps);
    val str = baos.toString();
    ps.close();
    str;
  }
}





