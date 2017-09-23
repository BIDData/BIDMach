package BIDMach.viz;
import BIDMat.{BMat,Mat,SBMat,CMat,DMat,FMat,FFilter,IMat,HMat,GDMat,GFilter,GLMat,GMat,GIMat,GSDMat,GSMat,LMat,SMat,SDMat,TMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import BIDMach.models.Model;
import BIDMach.networks.Net;
import BIDMach.networks.layers._;

/***
    Visualizing the filters within a ConvLayer
**/
class FilterViz(val layerId:Int, val bw:Int = 1, val name: String = "") extends Visualization{
    val _name = if (name.length > 0) name else "Conv@" + layerId
    val plot = new Plot(_name)
        
    override def check(model:Model, mats:Array[Mat]) = {
        model match {
            case net:Net => {
                if (layerId < net.layers.length){
                    net.layers(layerId) match {
                        case _:ConvLayer => 1;
                        case _=>{
                            println("The %d layer of the network is not a ConvLayer" format layerId);
                            2                                
                        }
                    }
                }
                else {
                    println("The network only has %d layers while you're accessing %d" format (net.layers.length, layerId))
                    2
                }                    
            }
            case _=>{
                println("The model is not a Net");
                2
            } 
        }                
    }
       
    override def doUpdate(model:Model, mats:Array[Mat], ipass:Int, pos:Long) = {
        val net = model.asInstanceOf[Net];
        val layer = net.layers(layerId).asInstanceOf[ConvLayer];
        val img = utils.filter2img(cpu(net.modelmats(layer.imodel)),net.opts.tensorFormat,bw);
        if (layer.imodel == 0){
            val dims = layer.output.dims
            val data = if (net.opts.tensorFormat == Net.TensorNHWC) layer.output 
                       else layer.output.asInstanceOf[FMat].fromNCHWtoNHWC
            val (d,i) = maxi2(data.reshapeView(dims(0),dims(1)*dims(2)*dims(3)),2);
            val id = IMat(i);
            val w = irow(id.data.map(_%dims(1)));  
            val h = irow(id.data.map(_%(dims(1)*dims(2))/dims(1)));  
            val n = irow(id.data.map(_/(dims(1)*dims(2))));
            val ffilter = layer.ffilter
            val ww = -ffilter.pad(1) + (w+ffilter.outPad(1)) * ffilter.stride(1)
            ww~ww+(ww<0)
            val hh = -ffilter.pad(2) + (h+ffilter.outPad(2)) * ffilter.stride(2)
            hh~hh+(hh<0)
            val res = FMat.zeros(irow(ffilter.inDims(1),ffilter.inDims(2),ffilter.inDims(0),dims(0)))
            val offset = net.layers(1).output.reshapeView(256,256,3,net.layers(1).output.dims(3))
            val bl = net.layers(4) match {
                case l:CropLayer=>l.blockInds
                case l:CropMirrorLayer=>l.blockInds
            }
            val idata = cpu(layer.inputData.reshapeView(layer.inputData.dims(1),layer.inputData.dims(2),
                                                         layer.inputData.dims(0),layer.inputData.dims(3))+
                             offset(bl(1),bl(2),?,?)
                            )
            for(k<-0 until dims(0)){
                res(?,?,?,k) = idata(ww(k)->(ww(k)+ffilter.inDims(1)),hh(k)->(hh(k)+ffilter.inDims(2)),?,n(k))                 
            }
            val input = utils.filter2img((res/256f-0.5f).reshapeView(ffilter.inDims(0),ffilter.inDims(1),ffilter.inDims(2),dims(0)),net.opts.tensorFormat,bw);
            plot.plot_image(img on input)
        }
        else
            plot.plot_image(img)
    }
    
}
