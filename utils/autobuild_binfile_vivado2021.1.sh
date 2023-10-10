#!/bin/bash
# $1 usr_name for server
# $2 ip_addr for server
# $3 is upload folder from your pc in elasticai.creator/build folder
# $4 is download folder to your device

# Clear auto_build folder
ssh $1@$2 "rm -rf /home/$1/.autobuild/*;exit;"

# Copy build folder from your device to server
scp -r $3 $1@$2:/home/$1/.autobuild/input_srcs

# Let vivado run
ssh $1@$2 "export XILINXD_LICENSE_FILE=/opt/flexlm/Xilinx.lic&&/tools/Xilinx/Vivado/2021.1/bin/vivado -mode tcl -source /home/$1/.autobuild_script/create_project_full_run.tcl;exit;"

# Copy *bin file to output folder
ssh $1@$2 "mkdir /home/$1/.autobuild/output; cp /home/$1/.autobuild/vivado_project/project_1.runs/impl_1/*.bin /home/$1/.autobuild/output/;exit;"

# Copy script in folder
ssh $1@$2 "mkdir /home/$1/.autobuild/tcl_script; cp /home/$1/.autobuild_script/create_project_full_run.tcl /home/$1/.autobuild/tcl_script/;exit;"

# Copy folders back to your machine
scp -r $1@$2:/home/$1/.autobuild/ $4

# Clear auto_build folder
ssh $1@$2 "rm -rf /home/$1/.autobuild/*;exit;"
