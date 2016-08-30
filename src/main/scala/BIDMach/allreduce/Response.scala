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


class Response(val ctype:Int, val dest0:Int, val clen:Int, val bytes:Array[Byte]) {
  val magic = Response.magic;
  var dest = dest0;
  val byteData = ByteBuffer.wrap(bytes);
  val intData = byteData.asIntBuffer;
  val floatData = byteData.asFloatBuffer;
  val longData = byteData.asLongBuffer;
  
  def encode() = {}
  def decode() = {}
  
  def this(ctype0:Int, dest0:Int, clen0:Int) = this(ctype0, dest0, clen0, new Array[Byte](4*clen0));
  
  override def toString():String = {
    "Response %s, length %d bytes" format (Response.names(ctype), clen*4);
  }
  
}

object Response {
	val magic = 0xa6b38734;
	final val configCtype = 1;
	final val permuteCtype = 2;
	final val allreduceCtype = 3;
	final val permuteAllreduceCtype = 4;
	final val setMachineCtype = 5;
	final val startLearnerCtype = 6;
	final val names = Array[String]("", "config", "permute", "allreduce", "permuteAllreduce", "setMachine", "startLearner");
	
	  
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





