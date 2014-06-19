#!/bin/bash

scalac -cp $ALL_LIBS Twitter.scala
scalac -cp $ALL_LIBS Yahoo.scala
scalac relabelmachines.scala
scalac splitmachines.scala






