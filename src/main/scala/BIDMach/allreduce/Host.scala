package BIDMach.allreduce

import BIDMat.{Mat,SBMat,CMat,DMat,FMat,IMat,HMat,GDMat,GLMat,GMat,GIMat,GSDMat,GSMat,LMat,SMat,SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
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
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.PrintStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.FileReader;
import java.io.BufferedReader;
import scala.collection.mutable.ArrayBuffer;
import java.io.IOException;


import java.io.IOException;


class Host(val opts:Host.Opts = new Host.Options) extends Serializable {

  var M = 0;
  var imach = 0;
  var round = 0;
  var gmods:IMat = null;
  var gridmachines:IMat = null;
  var masterIP:InetAddress = null;
  var workers:Array[InetSocketAddress] = null;
  var groups:Groups = null;
  var executor:ExecutorService = null;

  def log(msg:String) {
    print(msg);
  }

  class TimeoutThread(mtime:Int, futures:Array[Future[_]]) extends Runnable {
    def run() {
      try {
	Thread.sleep(mtime);
	for (i <- 0 until futures.length) {
	  if (futures(i) != null) {
	    if (opts.trace > 0) log("Worker cancelling thread %d" format i);
	    futures(i).cancel(true);
	  }
	}
      } catch {
	case e:InterruptedException => if (opts.trace > 2) log("Worker interrupted timeout thread");
      }
    }
  }
}

object Host {
  trait Opts extends BIDMat.Opts{
    var sendTimeout = 1000;
    var recvTimeout = 1000;
    var trace = 0;
    var machineTrace = 0;
    var commandSocketNum = 50050;
    var responseSocketNum = 50049;
    var peerSocketNum = 50051;
  }

  class Options extends Opts {}

  def inetIntToFields(v:Int):(Int,Int,Int,Int) = {
    val p0 = (v >> 24) & 255;
    val p1 = (v >> 16) & 255;
    val p2 = (v >> 8) & 255;
    val p3 = v & 255;
    (p0,p1,p2,p3)
  }

  def inetFieldsToInt(a:Int, b:Int, c:Int, d:Int):Int = {
    d + ((c + ((b + (a << 8)) << 8)) << 8);
  }

  def inetFieldsToString(p0:Int, p1:Int, p2:Int, p3:Int) = {
    "%d.%d.%d.%d" format(p0,p1,p2,p3);
  }

  def inetIntToString(v:Int) = {
    val (p0, p1, p2, p3) = inetIntToFields(v);
    "%d.%d.%d.%d" format(p0,p1,p2,p3);
  }

  def inetStringToFields(s:String):(Int, Int, Int, Int) = {
    println(s)
    val fields = s.split("\\.");
    val ff = fields.map(_.toInt);
    (ff(0), ff(1), ff(2), ff(3))
  }

  def inetStringToInt(s:String):Int = {
    val (p0,p1,p2,p3) = inetStringToFields(s);
    inetFieldsToInt(p0,p1,p2,p3)
  }

  def hostPortToInet(host:Int, port:Int):InetSocketAddress = {
    new InetSocketAddress(inetIntToString(host), port);
  }

  def hostPortsToInet(hosts:IMat, ports:IMat):Array[InetSocketAddress] = {
    val out = new Array[InetSocketAddress](hosts.length);
    for (i <- 0 until hosts.length) {
      out(i) = hostPortToInet(hosts(i), ports(i));
    }
    out;
  }

  def printStackTrace(e:Exception):String = {
    val baos = new ByteArrayOutputStream();
    val ps = new PrintStream(baos);
    e.printStackTrace(ps);
    val str = baos.toString();
    ps.close();
    str;
  }

  def getHosts(fname:String, defaultPort:Int = 50050):Array[InetSocketAddress] = {
    var i = 0;
    val ff = new BufferedReader(new FileReader(fname));
    var line = ff.readLine();
    val hosts = ArrayBuffer.empty[InetSocketAddress];
    while (line.length > 2) {
      val parts = line.split(":");
      val port = if (parts.length > 1) parts(1).toInt else defaultPort;
      hosts += new InetSocketAddress(parts(0), port);
      line = ff.readLine();
      i += 1;
    }
    ff.close();
    hosts.toArray;
  }

}

