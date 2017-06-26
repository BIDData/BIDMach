package BIDMach.networks.layers

import BIDMat.{Mat,DMat,FMat,FFilter,GMat,GDMat,GFilter}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import BIDMach.BIDMachSpec;
import BIDMach.networks.Net;
import org.scalatest._;
import org.scalatest.junit._;
import org.scalatest.prop._;
import org.junit.runner.RunWith

@RunWith(classOf[JUnitRunner])
class AddLayerTest extends BIDMachSpec {

    val w = 10;
    val h = 20;
    val c = 30;  
    val n = 40;
    
    val inmat1 = rand(w\h\c\n);
    val inmat2 = rand(w\h\c\n);
    val inderiv1 = zeros(w\h\c\n);
    val inderiv2 = zeros(w\h\c\n);
    val dout = rand(w\h\c\n);
    
    val input1 = Layer.input;
    val input2 = Layer.input;
    
    input1.output = inmat1;
    input2.output = inmat2;
    input1.deriv = inderiv1;
    input2.deriv = inderiv2;
    
    "An AddLayer" should "support forward computation" in {
      
      val net = new Net;
      val aa = input1 + input2;
      net.layers = Array(input1, input2, aa);
      net.forward;
      
      val cc = zeros(w\h\c\n);
      for (i <- 0 until cc.length) {
        cc.data(i) = inmat1.data(i) + inmat2.data(i);
      }	
    	checkSimilar(cc, aa.output.asInstanceOf[FMat]); 
    }

    it should "support backward computation" in {
      
    	val net = new Net;
      val aa = input1 + input2;
      net.layers = Array(input1, input2, aa);
      net.forward;
      
      aa.deriv should not be (null);
      
      aa.deriv <-- dout;
      net.backward(0,0);
      
    	checkSimilar(dout, input1.deriv.asInstanceOf[FMat]);
    	checkSimilar(dout, input2.deriv.asInstanceOf[FMat]); 
    }  

}
