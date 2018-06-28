package BIDMach.viz;
import BIDMat.{BMat,Mat,SBMat,CMat,DMat,FMat,FFilter,IMat,HMat,GDMat,GFilter,GLMat,GMat,GIMat,GSDMat,GSMat,LMat,SMat,SDMat,TMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import BIDMach.models.Model;
import BIDMach.networks.Net;
import BIDMach.networks.layers._;
import scala.collection.mutable.ListBuffer;

/***
    Visualizing some logged values
**/

abstract class LogViz(val name: String = "varName") extends Visualization{
    val plot = new ImageArray(name);
    val data:ListBuffer[FMat] = new ListBuffer[FMat];
    
    override def init(model:Model,mats:Array[Mat]) {

    }
    
    // Define this to collect some log data
    def collect():FMat
        
    override def doUpdate(model:Model, mats:Array[Mat], ipass:Int, pos:Long) = {  
      data.synchronized  {
    	  data += collect();
      }
    }
      
}