package BIDMach.viz;
import BIDMat.{BMat,Mat,SBMat,CMat,DMat,FMat,FFilter,IMat,HMat,GDMat,GFilter,GLMat,GMat,GIMat,GSDMat,GSMat,LMat,SMat,SDMat,TMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import BIDMach.models.Model;
import BIDMach.networks.Net;
import BIDMach.networks.layers._;

/***
    Visualizing the input and input gradient
    By default it uses guided backpropagation (See https://arxiv.org/pdf/1412.6806 for reference)
**/

class InputViz(val name: String = "Input") extends Visualization{
    val plot = new ImageArray(name);
    var deriv: Mat = null;
    var output: Mat = null;
    var guided_bp = true;
    var gradient_scale = 200f        
    var _gradient_scale: Mat = null;
    
    override def check(model:Model,mats:Array[Mat]) = {
        model match {
            case _:Net => 0;
            case _=>{println("Not a network"); 1}                
        }
    }
    
    //Creating deriv matries for InputLayer
    override def init(model:Model,mats:Array[Mat]) {
        val net = model.asInstanceOf[Net]
        for (i <- 0 until net.input_layers.length) {
            val layer = net.input_layers(i);
            if (layer.deriv.asInstanceOf[AnyRef] == null) 
                layer.deriv = layer.output.zeros(layer.output.dims);            
            layer.deriv.clear
            _gradient_scale = layer.deriv.ones(1,1);
        }
        net.opts.compute_input_gradient = true
    }
        
    override def doUpdate(model:Model, mats:Array[Mat], ipass:Int, pos:Long) = {        
        val net = model.asInstanceOf[Net]
        val layers = net.layers
        if (guided_bp) {
            var i = layers.length;
            while (i > 1) {
                i -= 1;
                if (layers(i).deriv.asInstanceOf[AnyRef] != null){
                    max(layers(i).deriv,0,layers(i).deriv);
                    layers(i).deriv ~ layers(i).deriv *@ (layers(i).output>=0)                    
                }
                layers(i).backward(ipass, pos);
            }
            net.updatemats.foreach(_ match{case m:Mat=>m.clear;case null=>})
        }
        val layer = layers(0).asInstanceOf[InputLayer];
        _gradient_scale(0,0) = gradient_scale
//        val img = utils.filter2img(layer.output(?,?,?,0->2)/256f-0.5f,net.opts.tensorFormat) \
//                  utils.filter2img(layer.deriv(?,?,?,0->2)*@_gradient_scale,net.opts.tensorFormat)
        val img = FMat(layer.output(?,?,?,0->2)) \ 
                  FMat((layer.deriv(?,?,?,0->2)*@_gradient_scale + 0.5f) * 256f)
        plot.plot_image(img, net.opts.tensorFormat)
        layer.deriv.clear
    }
}