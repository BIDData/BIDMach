package BIDMach.viz
import BIDMach.models.Model;
import BIDMat.Mat

abstract class Visualization {
    var interval = 10;
    var cnt = 0
    var checkStatus = 0 
        
    def doUpdate(model:Model,mats:Array[Mat],ipass:Int, pos:Long) = {}
    
    //Perform some initial check to make sure data type is correct
    def check(model:Model,mats:Array[Mat]):Int 
        
    def init(model:Model,mats:Array[Mat]) {}
       
    def update(model:Model,mats:Array[Mat],ipass:Int, pos:Long){
        if (checkStatus == 0){
            checkStatus = check(model, mats)
            if (checkStatus == 1) init(model, mats)
        }
        if (checkStatus == 1) {
            if (cnt == 0) {
                doUpdate(model, mats, ipass, pos)
                /*try { 
                    doUpdate(model, mats, ipass, pos)
                }
                catch {
                    case e:Exception=> {
                        checkStatus = 2
                        println(e.toString)
                    }
                }*/
            }
            cnt = (cnt + 1) % interval           
        }
    }
}

