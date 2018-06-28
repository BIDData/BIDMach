package BIDMach.viz;
import BIDMat.{BMat,Mat,SBMat,CMat,DMat,FMat,FFilter,IMat,HMat,GDMat,GFilter,GLMat,GMat,GIMat,GSDMat,GSMat,LMat,SMat,SDMat,TMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import BIDMach.models.Model;
import BIDMach.networks.Net;
import BIDMach.networks.layers._;
import BIDMach.Learner;
import scala.collection.mutable.ListBuffer;

/***
    Collect and Visualize some logged values
**/

class LogViz(val name: String = "varName") extends Visualization{
    val data:ListBuffer[FMat] = new ListBuffer[FMat];
    interval = 1;
    
    // Override one of these to collect some log data
    def collect(model:Model, mats:Array[Mat], ipass:Int, pos:Long):FMat = {
      collect(model);
    }
    
    def collect(model:Model):FMat = {
      collect();
    }
    
    def collect():FMat = {
      row(0);
    }
        
    override def doUpdate(model:Model, mats:Array[Mat], ipass:Int, pos:Long) = {  
      data.synchronized  {
    	  data += collect(model, mats, ipass, pos);
      }
    }
    
    def snapshot = {
      Learner.scores2FMat(data);
    }
      
}