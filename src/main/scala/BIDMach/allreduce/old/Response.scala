package BIDMach.allreduce.old

import BIDMat.{Mat,SBMat,CMat,DMat,FMat,IMat,HMat,GDMat,GLMat,GMat,GIMat,GSDMat,GSMat,LMat,SMat,SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import edu.berkeley.bid.comm._
import scala.collection.parallel._
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.net.ServerSocket;
import java.net.Socket;
import java.net.InetAddress;
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
import java.nio.charset.StandardCharsets

class Response (
  val rtype:Int, round0:Int, val src:Int, val clen:Int, val bytes:Array[Byte], val blen:Int,
  val tag:String = null
) {
  val magic = Response.magic;
  var round = round0;
  val byteData = ByteBuffer.wrap(bytes);
  val intData = byteData.asIntBuffer;
  val floatData = byteData.asFloatBuffer;
  val longData = byteData.asLongBuffer;
  
  def encode() = {}
  def decode() = {}
  
  def this(rtype0:Int, round0:Int, dest0:Int, clen0:Int, tag:String) =
    this(rtype0, round0, dest0, clen0, new Array[Byte](4*clen0), 4*clen0, tag);

  def this(rtype0:Int, round0:Int, dest0:Int, clen0:Int) =
    this(rtype0, round0, dest0, clen0, new Array[Byte](4*clen0), 4*clen0, null);
  
  def this(rtype0:Int, round0:Int, dest0:Int, tag:String) =
    this(rtype0, round0, dest0, 0, new Array[Byte](0), 0, tag);

  def this(rtype0:Int, round0:Int, dest0:Int) =
    this(rtype0, round0, dest0, 0, new Array[Byte](0), 0, null);
  
  override def toString():String = {
    "Response %s, tag %s, round %d, src %d, length %d bytes" format (
      Command.names(rtype), tag, round, src, clen*4);
  }
  
}

class AllreduceResponse(round0:Int, src0:Int, var success:Boolean, bytes:Array[Byte], tag:String)
extends Response(Command.allreduceCtype, round0, src0, 1, bytes, 1*4, tag) {

  def this(round0:Int, src0:Int, success:Boolean, tag:String) =
    this(round0, src0, success, new Array[Byte](1*4), tag)

  override def encode():Unit = {
    intData.rewind()
    intData.put(if (success) 1 else 0)
  }

  override def decode():Unit = {
    intData.rewind()
    success = (intData.get() == 1)
  }
}
object AllreduceResponse {
  def success(round:Int, src:Int, tag:String) = {
    val resp = new AllreduceResponse(round, src, true, tag)
    resp.encode
    resp
  }
  def failure(round:Int, src:Int, tag:String) = {
    val resp = new AllreduceResponse(round, src, false, tag)
    resp.encode
    resp
  }
}


class AckReadyResponse(round0:Int, src0:Int, bytes:Array[Byte])
extends Response(Command.ackReadyCtype, round0, src0, 1, bytes, 1*4) {

  def this(src0:Int) = this(-1, src0, new Array[Byte](1*4));

  override def encode():Unit = {
    intData.rewind();
    intData.put(src);
  }

  override def decode():Unit = {}
}


class RegisterWorkerResponse(workerIP0:InetAddress, workerCmdPort0:Int, bytes:Array[Byte])
extends Response(Command.registerWorkerCtype, -1, 0, 1, bytes, 2*4) {

  var workerIP:InetAddress = workerIP0
  var workerCmdPort:Int = workerCmdPort0

  def this(bytes:Array[Byte]) =
    this(null, -1, bytes)

  def this(workerCmdPort0:Int) =
    this(InetAddress.getLocalHost, workerCmdPort0, new Array[Byte](2*4))

  override def encode():Unit = {
    intData.rewind()
    intData.put(Host.inetStringToInt(workerIP.getHostAddress))
    intData.put(workerCmdPort)
  }

  override def decode():Unit = {
    intData.rewind()
    val intWorkerIP = intData.get()
    workerIP = InetAddress.getByName(Host.inetIntToString(intWorkerIP))
    workerCmdPort = intData.get()
  }

  def workerCmdSocketAddr:InetSocketAddress = {
    return new InetSocketAddress(workerIP, workerCmdPort)
  }
}


class ReturnObjectResponse(round0:Int, src0:Int, obj0:AnyRef, bytes:Array[Byte])
extends Response(Command.returnObjectCtype, round0, src0, bytes.size, bytes, bytes.size) {

  var obj:AnyRef = obj0;

  def this(round0:Int, dest0:Int, obj0:AnyRef) = {
    this(round0, dest0, obj0, {
      val out  = new ByteArrayOutputStream()
      val output = new ObjectOutputStream(out)
      output.writeObject(obj0)
      output.close
      out.toByteArray()
    });
  }
  
  override def encode ():Unit = { }
  
  override def decode():Unit = {    
		val in = new ByteArrayInputStream(bytes);
		val input = new ObjectInputStream(in);
		obj = input.readObject;
		input.close;
  }
}

class WorkerExceptionResponse(round0:Int, src0:Int, var msg:String, bytes:Array[Byte])
extends Response(Command.workerExceptionCtype, round0, src0, 4+msg.length, bytes, 4+msg.length) {

  def this(round0:Int, src0:Int, msg0:String) =
    this(round0, src0, msg0, new Array[Byte](4+msg0.length))

  override def encode():Unit = {
    intData.rewind()
    byteData.rewind()
    val msgBytes = msg.getBytes(StandardCharsets.UTF_8)
    intData.put(msgBytes.length)
    byteData.position(4)
    byteData.put(msgBytes, 0, msgBytes.length)
  }

  override def decode():Unit = {
    intData.rewind()
    byteData.rewind()
    val msgLength = intData.get()
    val msgBytes = new Array[Byte](msgLength)
    byteData.position(4)
    byteData.get(msgBytes, 0, msgLength)
    msg = new String(msgBytes, StandardCharsets.UTF_8)
  }
}

class ResponseWriter(address:InetSocketAddress, resp:Response, me:Worker)
extends Runnable {
  def run() {
    var socket:Socket = null;
    try {
      socket = new Socket();
      socket.setReuseAddress(true);
      socket.connect(address, me.opts.sendTimeout);
      if (socket.isConnected()) {
        val ostr = new DataOutputStream(socket.getOutputStream());
        ostr.writeInt(resp.magic)
        ostr.writeInt(resp.rtype);
        ostr.writeInt(resp.round);
        ostr.writeInt(resp.src);
	if (resp.tag != null) {
	  ostr.writeInt(resp.tag.length)
          val tagBytes = resp.tag.getBytes(StandardCharsets.UTF_8)
	  ostr.write(tagBytes, 0, resp.tag.length)
	} else {
	  ostr.writeInt(0)
	}
        ostr.writeInt(resp.clen);
        ostr.writeInt(resp.blen);
        ostr.write(resp.bytes, 0, resp.blen);
      }
    } catch {
      case e:Exception => if (me.opts.trace > 0) {
        me.log("Master problem sending resp %s\n%s\n" format (resp.toString, Response.printStackTrace(e)));
      }
    } finally {
      try {
        if (socket != null) socket.close();
      } catch {
        case e:Exception => if (me.opts.trace > 0) me.log("Master problem closing socket\n%s\n" format Response.printStackTrace(e));
      }
    }
  }
}

class ResponseReader(socket:Socket, me:Master) extends Runnable {
    def run() {
      try {
        val istr = new DataInputStream(socket.getInputStream());
        val magic = istr.readInt();
        val rtype = istr.readInt();
        val round = istr.readInt();
        val dest = istr.readInt();
	val tagLen = istr.readInt()
	var tag:String = null
        if (tagLen > 0) {
          val tagBuf = new Array[Byte](tagLen)
          istr.readFully(tagBuf, 0, tagLen)
          tag = new String(tagBuf, StandardCharsets.UTF_8)
        }
        val clen = istr.readInt();
        val blen = istr.readInt();
        val response = new Response(rtype, round, dest, clen, new Array[Byte](blen), blen, tag);
        if (me.opts.trace > 2) me.log("Master got packet %s\n" format (response.toString));
        istr.readFully(response.bytes, 0, blen);

        try {
          socket.close();
        } catch {
          case e:IOException => {if (me.opts.trace > 0) me.log("Master Problem closing socket "+Response.printStackTrace(e)+"\n")}
        }
        me.handleResponse(response);
      } catch {
      case e:Exception => if (me.opts.trace > 0) me.log("Master Problem reading socket "+Response.printStackTrace(e)+"\n");
      } finally {
        try {
          if (!socket.isClosed) socket.close();
        } catch {
        case e:IOException => {if (me.opts.trace > 0) me.log("Master Final Problem closing socket "+Response.printStackTrace(e)+"\n")}
        }
      }
    }
  }

object Response {
	val magic = 0xa6b38734;
  
  def printStackTrace(e:Exception):String = {
    val baos = new ByteArrayOutputStream();
    val ps = new PrintStream(baos);
    e.printStackTrace(ps);
    val str = baos.toString();
    ps.close();
    str;
  }
}
