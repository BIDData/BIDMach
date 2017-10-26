package BIDMach.viz
/**
**/

import BIDMat.{BMat,Mat,SBMat,CMat,DMat,FMat,FFilter,IMat,HMat,GDMat,GFilter,GLMat,GMat,GIMat,GSDMat,GSMat,LMat,SMat,SDMat,TMat,Image}
import BIDMat.MatFunctions._
import BIDMat.MatFunctions
import BIDMat.SciFunctions._
import java.util.concurrent.Future;
import BIDMach.Learner
import BIDMach.networks.Net
import BIDMach.networks.layers._
import javax.swing._;
import javax.swing.event._
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

class Plot(name: String = "plot") {
    val id = {Plot.tot+=1;Plot.tot}
    val frame = new JFrame(name)
    var img: BufferedImage = null
    var server = {if (Plot.server == null) Plot.server = new WebServer(); Plot.server}
    val imgPanel = new JPanel();frame.getContentPane().add(imgPanel)
    val controlPanel = new JPanel();frame.getContentPane().add(controlPanel)
    frame.getContentPane().setLayout(new BoxLayout(frame.getContentPane(),BoxLayout.PAGE_AXIS))        
    controlPanel.setLayout(new BoxLayout(controlPanel,BoxLayout.PAGE_AXIS))        
        
    def plot_image(data:IMat) {
        if (img == null || img.getWidth != data.ncols || img.getHeight != data.nrows){
            img = new BufferedImage(data.ncols,data.nrows,BufferedImage.TYPE_INT_ARGB)
            imgPanel.removeAll
            imgPanel.add(new JLabel(new ImageIcon(img)))
            frame.pack()
            frame.setVisible(true);
        }
        val buf=img.getRaster().getDataBuffer().asInstanceOf[java.awt.image.DataBufferInt].getData()
        val datat = data.t;
        Array.copy(datat.data,0,buf,0,buf.length);
        frame.repaint();
        plot_web(data)
    }
    
    def plot_web(data:IMat) {
        //server.send(scala.util.parsing.json.JSONObject())
    }    
    
    def add_slider(name:String,callback:Int=>Float,initV:Int = 50, precision:Int = 2) {
        val p = new JPanel();
        val sliderLabel = new JLabel(name, SwingConstants.CENTER);
        val rangeLabel = new JLabel(("%."+precision+"f") format callback(initV), SwingConstants.CENTER);
        val slider = new JSlider(SwingConstants.HORIZONTAL,0,100,initV);
        slider.addChangeListener(new ChangeListener{
                                    override def stateChanged(e:ChangeEvent){
                                        val source = e.getSource().asInstanceOf[JSlider];
                                        val v = callback(source.getValue());
                                        rangeLabel.setText(("%."+precision+"f") format v)
                                    }})
        p.add(sliderLabel);
        p.add(slider);
        p.add(rangeLabel);
        controlPanel.add(p)
        frame.pack                        
    }
}

object Plot{
    var tot = 0
    var interval = 2
    val useWeb = false
    var currentLearner: Learner = null 
    var currentNet: Net = null   
    val d3category10 = Array(2062260, 16744206, 2924588, 14034728, 9725885, 9197131, 14907330, 8355711, 12369186, 1556175)
    var server: WebServer = null    
    //https://github.com/d3/d3-scale/blob/master/README.md#schemeCategory10
    //val color=Array("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf")

    var tensorFormat = Net.TensorNCHW;
    var gradient_scale = 5000f
    
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

    def plot_loss(nn:Learner,group: Int = 10) = {
        Plotting.plot(()=>{
            val res = nn.reslist.toArray.map(_.dv.toFloat).grouped(group).map(x=>x.reduce(_+_)/x.length).toArray
            FMat(Array(1,res.length),res)
        })
    }
    
    def plot_layer(layerId: Int) = {
        var layer = currentNet.layers(layerId)
        val layerName = layer.getClass.getSimpleName
        val bidmachURL = "https://raw.githubusercontent.com/BIDData/BIDMach/master/src/main/"
        if (layerName == "ConvLayer"){
            currentLearner.add_plot(new FilterViz(layerId))
        } else if (layerName == "InputLayer") {
            val v = currentLearner.add_plot(new InputViz()).asInstanceOf[InputViz];
            v.guided_bp =true                
        }
        else plot_code(bidmachURL + "scala/BIDMach/networks/layers/" + layerName + ".scala")
    }
}
