package BIDMach.viz;
import BIDMat.{BMat,Mat,SBMat,CMat,DMat,FMat,FFilter,IMat,HMat,GDMat,GFilter,GLMat,GMat,GIMat,GSDMat,GSMat,LMat,SMat,SDMat,TMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import BIDMach.models.Model;
import BIDMach.networks.Net;
import BIDMach.networks.layers._;

/***
    Synthesizing the image
**/

class Synthesis(val name: String = "Input") extends Visualization{
    val plot = new Plot(name);
    var lrate = 0.1f;
    var _lrate: Mat = null;
    
    def check(model:Model, mats:Array[Mat]) = 1
       
    override def init(model:Model, mats:Array[Mat]) = {
        interval  = 100
        val net = model.asInstanceOf[Net]
        for (i <- 0 until net.input_layers.length) {
            val layer = net.input_layers(i);
            if (layer.deriv.asInstanceOf[AnyRef] == null) 
                layer.deriv = layer.output.zeros(layer.output.dims);            
            layer.deriv.clear;
            _lrate = layer.deriv.zeros(1,1);
                
        }        
    }

    override def doUpdate(model:Model, mats:Array[Mat], ipass:Int, pos:Long) = {        
        val net = model.asInstanceOf[Net];
        val layer = net.layers(0).asInstanceOf[InputLayer];
        val srcImg = utils.filter2img(layer.output(?,?,?,0->2)/256f-0.5f,net.opts.tensorFormat);
        _lrate(0,0) = lrate;
        for(t<-0 until 10){
            net.forward;
            net.setderiv()
            net.backward(ipass, pos);
            layer.output~layer.output + (layer.deriv *@ _lrate)
        }
        
        val img = srcImg \ utils.filter2img(layer.output(?,?,?,0->2)/256f-0.5f,net.opts.tensorFormat)
        plot.plot_image(img)
    }    
}
    
    
    