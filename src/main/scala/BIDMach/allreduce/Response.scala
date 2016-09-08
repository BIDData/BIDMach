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


class Response(val rtype:Int, round0:Int, val src:Int, val clen:Int, val bytes:Array[Byte]) {
  val magic = Response.magic;
  var round = round0;
  val byteData = ByteBuffer.wrap(bytes);
  val intData = byteData.asIntBuffer;
  val floatData = byteData.asFloatBuffer;
  val longData = byteData.asLongBuffer;
  
  def encode() = {}
  def decode() = {}
  
  def this(rtype0:Int, round0:Int, dest0:Int, clen0:Int) = this(rtype0, round0, dest0, clen0, new Array[Byte](4*clen0));
  
  def this(rtype0:Int, round0:Int, dest0:Int) = this(rtype0, round0, dest0, 0, null);
  
  override def toString():String = {
    "Response %s, round %d, src %d, length %d bytes" format (Command.names(rtype), round, src, clen*4);
  }
  
}

class ResponseWriter(dest:String, socketnum:Int, resp:Response, me:Worker) extends Runnable {

	def run() {
		var socket:Socket = null;
	try {
		socket = new Socket();
		socket.setReuseAddress(true);
		socket.connect(new InetSocketAddress(dest, socketnum), me.opts.sendTimeout);
		if (socket.isConnected()) {
			val ostr = new DataOutputStream(socket.getOutputStream());
			ostr.writeInt(resp.magic)
			ostr.writeInt(resp.rtype);
      ostr.writeInt(resp.round);
			ostr.writeInt(resp.src);
			ostr.writeInt(resp.clen);
			ostr.write(resp.bytes, 0, resp.clen*4);   
		}
	} catch {
	case e:Exception =>
	if (me.opts.trace > 0) {
		me.log("Master problem sending resp %s\n%s\n" format (resp.toString, Response.printStackTrace(e)));
	}
	} finally {
		try { if (socket != null) socket.close(); } catch {
		case e:Exception =>
		if (me.opts.trace > 0) me.log("Master problem closing socket\n%s\n" format Response.printStackTrace(e));        
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
        val clen = istr.readInt();
        val response = new Response(rtype, round, dest, clen, new Array[Byte](clen*4));
        if (me.opts.trace > 2) me.log("Master got packet %s\n" format (response.toString));
        istr.readFully(response.bytes, 0, clen*4);
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





