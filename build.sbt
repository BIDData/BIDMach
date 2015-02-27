
name := "BIDMach"

version := "1.0.0"

organization := "edu.berkeley.bid"

scalaVersion := "2.11.2"

resolvers ++= Seq(
  "Scala Tools Snapshots" at "http://scala-tools.org/repo-snapshots/",
  "Scala Mirror" at "https://oss.sonatype.org/content/repositories/releases/"
)

libraryDependencies <<= (scalaVersion, libraryDependencies) { (sv, deps) =>
  deps :+ ("org.scala-lang" % "scala-compiler" % sv)
}

libraryDependencies += "org.scala-lang" % "jline" % "2.10.3"

libraryDependencies += "org.apache.commons" % "commons-math3" % "3.2"

//libraryDependencies += "org.scalatest" %% "scalatest" % "2.0" % "test"

//libraryDependencies += "org.scalacheck" %% "scalacheck" % "1.11.2" % "test"

libraryDependencies += "junit" % "junit" % "4.5" % "test"

// Add these some day but for now they're missing native code dedendencies
//libraryDependencies += "net.jpountz.lz4" % "lz4" % "1.3"

//libraryDependencies += "org.scala-saddle" % "jhdf5" % "2.9"

credentials += Credentials(Path.userHome / ".ivy2" / ".credentials")

javacOptions ++= Seq("-source", "1.7", "-target", "1.7")

scalacOptions ++= Seq("-deprecation","-target:jvm-1.7")

initialCommands := scala.io.Source.fromFile("lib/bidmach_init.scala").getLines.mkString("\n")

javaOptions += "-Xmx12g"




