package BIDMach.viz
import BIDMach.models.Model;
import BIDMat.Mat


/**
    Abstract class for visualizations. Extend this class to get correct behavior 
*/

abstract class Visualization {
    var interval = 10;
    var cnt = 0
    var checkStatus = -1
        
    def doUpdate(model:Model,mats:Array[Mat],ipass:Int, pos:Long)
    
    //Perform some initial check to make sure data type is correct
    def check(model:Model,mats:Array[Mat]):Int = 0
        
    //Initialize variables and states during the first update.
    def init(model:Model,mats:Array[Mat]) {}
       
    //Update the visualization per cnt batches
    def update(model:Model,mats:Array[Mat],ipass:Int, pos:Long){
        if (checkStatus == -1){
            checkStatus = check(model, mats)
            if (checkStatus == 0) init(model, mats)
        }
        if (checkStatus == 0) {
            if (cnt == 0) {
                //doUpdate(model, mats, ipass, pos)
                try { 
                    doUpdate(model, mats, ipass, pos)
                }
                catch {
                    case e:Exception=> {
                        checkStatus = 2
                        println(e.toString)
                        println(e.getStackTrace.mkString("\n"))
                    }
                }
            }
            cnt = (cnt + 1) % interval           
        }
    }
}
