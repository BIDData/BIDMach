package BIDMach.viz

import BIDMat.{BMat,Mat,SBMat,CMat,DMat,FMat,FFilter,IMat,HMat,GDMat,GFilter,GLMat,GMat,GIMat,GSDMat,GSMat,LMat,SMat,SDMat,TMat,Image}
import BIDMat.MatFunctions._
import BIDMat.MatFunctions
import BIDMat.SciFunctions._
import java.util.concurrent.Future;
import BIDMach.Learner
import BIDMach.networks.Net
import BIDMach.networks.layers._
import javax.swing._;
import com.mxgraph.swing.mxGraphComponent;
import com.mxgraph.view.mxGraph;
import scala.collection.mutable.HashMap
import java.awt.event._
import java.awt.image.BufferedImage
import ptolemy.plot._
import BIDMat.{Plotting,MyPlot}


class GraphFrame(val layers:Array[HashMap[String,Any]])extends JFrame("Graph"){
    val graph = new mxGraph();
    val _parent = graph.getDefaultParent();
    graph.getModel().beginUpdate();
    var i = 0
    val nodes = layers.map(d=>{
        graph.insertVertex(_parent, null, d("layerId") + " " + d("name"), 20, 20 + 40*i, 200,30)
        i+=1
    })
        //            graph.insertEdge(parent2, null, "Edge", v1, v2);
    graph.getModel().endUpdate();

    val graphComponent = new mxGraphComponent(graph) {
        override def installDoubleClickHandler(){
            getGraphControl()
                .addMouseListener(new MouseAdapter(){
                    override def mouseClicked(e: MouseEvent){
                        if(e.getClickCount()==2){
                            val cell = getCellAt(e.getX(), e.getY());
                            if (cell != null)
                                Plot.plot_layer(graph.getLabel(cell).toString.split(" ")(0).toInt)//.split("\"")(1));
                        }
                    }
                })
        }
    }
    getContentPane().add(graphComponent);  
    graphComponent.getGraphControl()
        .addMouseListener(new MouseAdapter()
                          {
                              override def mouseReleased(e:MouseEvent)
                              {
                                  val  cell = graphComponent.getCellAt(e.getX(), e.getY());

                                  if (cell != null)
                                  {
                                      //println("cell="+graph.getLabel(cell));
                                  }
                              }                              
                          }); 
}

object Plot_active{
    var tot = 0
    var interval = 2
    val useWeb = false
    var currentLearner: Learner = null 
    var currentNet: Net = null   
    val d3category10 = Array(2062260, 16744206, 2924588, 14034728, 9725885, 9197131, 14907330, 8355711, 12369186, 1556175)
    var tensorFormat = Net.TensorNCHW;
    var gradient_scale = 5000f
    
    //https://github.com/d3/d3-scale/blob/master/README.md#schemeCategory10
    //val color=Array("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf")
        
    def getMsg(id:String, ipass:Int, names:List[String], values:Array[Float]) = {
         "{\"id\":\"%s\",\"ipass\":%d,\"names\":[%s],\"values\":[%s]}" format(id,ipass,names.map('"'+_+'"').reduce(_+","+_),values.map('"'+_.toString+'"').reduce(_+","+_))                    
    }
        
    def plot(fn:()=>Mat,names:List[String]=List("data")) {
        /**if (server == null)
            start()
        tot += 1
        val id = "Plot_" + tot
        var ipass = 0
        val runme = new Runnable {
          override def run() = {
            while (true) {
                Thread.sleep((1000*interval).toLong);
                val mat = MatFunctions.cpu(fn());
                ipass += 1
                val message = getMsg(id,ipass,names,FMat(mat).data)
                server.send(message)
//                println(message)
            }
          }
        }
        Image.getService.submit(runme);*/
    }
    
    def plot_web(learner:Learner) = {        
        /**
        if (server == null)
            start()
        server.send(utils.getModelGraph(learner)._2.toString)*/
        null
    }
    
    def plot_jframe(learner:Learner) = {
        val obj = utils.getModelGraph(learner)
        val layers = obj("layers").asInstanceOf[Array[HashMap[String,Any]]]
        val frame = new GraphFrame(layers)
        //frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setSize(800, 600);
        frame.setVisible(true);
        frame
    }
    
    def plot(learner:Learner):GraphFrame = {
        currentLearner = learner
        currentNet = learner.model.asInstanceOf[Net]
        if (useWeb) plot_web(learner)
        else plot_jframe(learner)
    }

    def plot_code(filename:String) = {
        val title = filename.split("/").reverse.head
        val frame = new JFrame(title);
        val textArea = new JTextArea(60, 40);
        val scrollPane = new JScrollPane(textArea);
        val s=scala.io.Source.fromURL(filename).getLines.mkString("\n");
        textArea.append(s)        
        frame.getContentPane().add(scrollPane)
        //frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setSize(800, 600);
        frame.setVisible(true);
        frame
    }

    def plot_image(fn:()=>IMat,name:String) = {
        val p = new MyPlot;
        val data = fn()
        p.frame = new PlotFrame(name, p);
        val img = new BufferedImage(data.ncols,data.nrows,BufferedImage.TYPE_INT_ARGB)
        val buf=img.getRaster().getDataBuffer().asInstanceOf[java.awt.image.DataBufferInt].getData()
        val datat = data.t
        Array.copy(datat.data,0,buf,0,buf.length)
        p.frame.getContentPane().add(new JLabel(new ImageIcon(img)))
        p.frame.pack()
        p.frame.setVisible(true);
        p.done = false;
        val runme = new Runnable {
            override def run() = {
                while (!p.done) {
                    val data = fn()
                    Array.copy(data.t.data,0,buf,0,buf.length)
                    p.frame.repaint()
                    Thread.sleep((1000*interval).toLong);                
                }
            }
        }
        p.fut = Image.getService.submit(runme);
        p
    }
    
    def getV(f:Double) = {
        val d = (f*256).toInt+128
        if (d>255) 255 else d
    }
    

    def getFilterImg(data:Mat,bw:Int = 1) = {
        val in_channel = data.dims(0)
        val w = data.dims(1)
        val h = data.dims(2)
        val num = data.dims(3)
        val new_data = (data.reshape(Array(w,h,in_channel,num))).asInstanceOf[FMat]
        val col = Math.sqrt(num).toInt
        val row = Math.ceil(num*1f/col).toInt
        val out = IMat(row*h*bw,col*w*bw)
        var i = 0
        var j = 0
        for(k<-0 until num){
            for(r<-0 until h)
                for(c<-0 until w){
                    val ii = i*h+r
                    val jj = j*w+c
                    //println(i,j,ii,jj,out.nrows,out.ncols)
                        
                    val ind = c+w*r+k*w*h*3
                    var res = getV(new_data.data(ind))
                    res += getV(new_data.data(ind+w*h))*256
                    res += getV(new_data.data(ind+w*h*2))*256*256
                    res += 255*256*256*256
                        
                    /*var res  = getV(new_data(c,r,0,k).dv)
                    res += getV(new_data(c,r,1,k).dv)*256
                    res += getV(new_data(c,r,2,k).dv)*256*256
                    res += 255*256*256*256*/
                    /*var res  = ((data(0,r,c,k)+0.5f).dv*256).toInt
                    res += ((data(1,r,c,k)+0.5f).dv*256).toInt*256
                    res += ((data(2,r,c,k)+0.5f).dv*256).toInt*256*256
                    res += 255*256*256*256*/
                    out(ii*bw->(ii*bw+bw),jj*bw->(jj*bw+bw)) = res
                }
            j+=1
            if (j == col) {j = 0;i+=1}
        }
        out
    }

    def plot_filters(fn:()=>Mat,name:String = "conv",bw:Int = 10) {
        plot_image(
            ()=>getFilterImg(MatFunctions.cpu(fn()),bw),
            name)    
    }

    def plot_loss(nn:Learner,group: Int = 10) = {
        Plotting.plot(()=>{
            val res = nn.reslist.toArray.map(_.dv.toFloat).grouped(group).map(x=>x.reduce(_+_)/x.length).toArray
            FMat(Array(1,res.length),res)
        })
    }
    
    def plot_input(data:Mat,name:String) {
        plot_image(
            ()=>{
                val idata = MatFunctions.cpu(data)
                getFilterImg(idata(?,?,?,0->4)/256f-0.5f)
            }
            ,name
        )
    }
    
    def plot_layer(layerId: Int) = {
        var layer = currentNet.layers(layerId)
        val layerName = layer.getClass.getSimpleName
        val bidmachURL = "https://raw.githubusercontent.com/BIDData/BIDMach/master/src/main/"
        if (layerName == "ConvLayer"){
            val cl = layer.asInstanceOf[ConvLayer]
            plot_filters(()=>{currentNet.modelmats(cl.imodel)},"Conv@"+layerId,2)
            if (cl.nearestImg.asInstanceOf[AnyRef]!=null)
                Plot.plot_filters(()=>cl.nearestImg.reshapeView(3,11,11,96)/256f-0.5f,"imgs",2);
        } else if (layerName == "InputLayer") {
            val il = layer.asInstanceOf[InputLayer]
//            val idata = MatFunctions.cpu()
            val buf = il.derivCPU.zeros((il.derivCPU(?,?,?,0->4)).dims)
            plot_image(()=>{
                buf(?,?,?,0->2) = il.outputCPU(?,?,?,0->2)/256f-0.5f;
                buf(?,?,?,2->4) = (il.derivCPU*gradient_scale)(?,?,?,0->2);
                getFilterImg(buf,1)
            },
            "InputGradient")                                
            /*plot_input(il.output,"Input")
            plot_image(
                ()=>getFilterImg((il.derivCPU*1000)(?,?,?,0->4),1),
                "InputGradient")*/
            //plot_filters(()=>(MatFunctions.cpu(il.deriv) * 300000)(?,?,?,0->4),"InputGradient",1)
        }
        else plot_code(bidmachURL + "scala/BIDMach/networks/layers/" + layerName + ".scala")
    }
}
