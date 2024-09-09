





(1), Provide stepwise list on prerequisite installation on the Server node

* Directory structure and set the common environmental variables
```bash
##########################################################################
# Directory structures on web server node:
#   /boot
#   /local/
#       NCI_HERE_app/
#           website/                
#           tmp/
#           install/
#   /mnt/
#       disks/
#           disk1/
#               HERE_assets/                    # Please make sure that the disk1 has 3TB+ free space
##########################################################################
APP_ROOT=/local/NCI_HERE_app                    # app root (CHANGE THIS)
WEB_ROOT=${APP_ROOT}/website                    # website root (CHANGE THIS)
HERE_DEPS_INSTALL=${APP_ROOT}/install           # dependencies installed
HERE_DEPS_TMP=${APP_ROOT}/tmp                   # dependencies source, build, etc.
DST_DATA_ROOT=/mnt/hidare-efs/data/HERE_assets      # data assets, e.g., images (3TB+ free space)
APP_USERNAME=zhongz2				  # CHANGE THIS
##########################################################################
# please make sure the operating user can read and write these directories on the web server (skip if already existed)
sudo mkdir -p ${DST_DATA_ROOT}/temp_images_dir ${APP_ROOT} ${HERE_DEPS_INSTALL} ${HERE_DEPS_TMP}
sudo chown ${APP_USERNAME}:${APP_USERNAME} -R ${APP_ROOT} ${HERE_DEPS_INSTALL} ${HERE_DEPS_TMP}  # optional
sudo chown ${APP_USERNAME}:${APP_USERNAME} ${DST_DATA_ROOT}						   # optional


export PATH=${HERE_DEPS_INSTALL}/bin:$PATH
export LD_LIBRARY_PATH=${HERE_DEPS_INSTALL}/lib64:${HERE_DEPS_INSTALL}/lib:$LD_LIBRARY_PATH
export PKG_CONFIG_PATH=${HERE_DEPS_INSTALL}/lib64/pkgconfig:${HERE_DEPS_INSTALL}/lib/pkgconfig:$PKG_CONFIG_PATH
```


* Install dependency libraries on CentOS 7

```bash
# Install packages on CentOS Linux release 7.9.2009 (Core)
# $uname -a
# Linux instance-20240626-214050 3.10.0-1160.119.1.el7.x86_64 #1 SMP Tue Jun 4 14:43:51 UTC 2024 x86_64 x86_64 x86_64 GNU/Linux
# Please find the yum_list.txt for a complete list of all installed packages
sudo yum install -y \
   openjpeg-devel libjpeg-turbo-devel openjpeg \
   libjpeg-turbo unzip tmux gobject-introspection-devel \
   gcc-c++ libgsf-devel libgsf libtiff-devel libtiff \
   gsl-devel gsl \
   httpd httpd-devel keyutils-libs-devel krb5-devel libkadm5 \
   rsync kernel-headers libcom_err-devel libsepol-devel \
   gcc cpp zlib-devel glibc-devel glibc-headers libverto-devel \
   pcre-devel openssl-devel bzip2-devel libffi-devel \
   wget make apr-devel cyrus-sasl cyrus-sasl-devel \
   openldap-devel apr-util-devel expat-devel libdb-devel \
   sqlite-devel gdbm-devel pyparsing systemtap-sdt-devel \
   xz-devel fontpackages-filesystem bzip2 \
   ncurses-devel readline libglvnd gdk-pixbuf2 jasper-libs \
   jbigkit-libs libglvnd-egl libglvnd-glx glib2-devel \
   libxml2-devel tk-devel openssl-devel readline-devel \
   gcc openssl-devel bzip2-devel libffi-devel zlib-devel \
   gcc zlib-devel bzip2 bzip2-devel readline-devel sqlite sqlite-devel \
   openssl-devel tk-devel libffi-devel xz-devel gdbm-devel ncurses-devel \
   wget git
# db4-devel openslide-devel openslide

# install OpenSSL 1.1.1
ImportError: urllib3 v2 only supports OpenSSL 1.1.1+, currently the 'ssl' module is compiled with 'OpenSSL 1.0.2k-fips  26 Jan 2017'. See: https://github.com/urllib3/urllib3/issues/2168
cd $HERE_DEPS_TMP
wget https://github.com/openssl/openssl/releases/download/OpenSSL_1_1_1s/openssl-1.1.1s.tar.gz
tar -xvf openssl-1.1.1s.tar.gz
cd openssl-1.1.1s
./config --prefix=${HERE_DEPS_INSTALL} --openssldir=${HERE_DEPS_INSTALL}/etc/ssl --libdir=lib no-shared zlib-dynamic
make
make install
```



* Install MySQL (skip if installed)
```bash
cd $HERE_DEPS_TMP
curl -sSLO https://dev.mysql.com/get/mysql84-community-release-el7-1.noarch.rpm
md5sum mysql84-community-release-el7-1.noarch.rpm
sudo rpm -ivh mysql84-community-release-el7-1.noarch.rpm
sudo yum install -y mysql-server
sudo systemctl start mysqld
```

* Install Python-3.9.18 (skip if installed)
```bash
# Compile Python (skip if already installed)
cd ${HERE_DEPS_TMP}
if [ ! -e Python-3.9.18.tgz ]; then wget https://www.python.org/ftp/python/3.9.18/Python-3.9.18.tgz; fi
if [ -d Python-3.9.18 ]; then rm -rf Python-3.9.18; fi
tar -xvf Python-3.9.18.tgz        
cd Python-3.9.18
CFLAGS=-fPIC ./configure --enable-shared=no --prefix=${HERE_DEPS_INSTALL}
make
make install
```


* Install Python virtual environment
```bash
# Virtual environment
cd ${APP_ROOT}
pip3 install virtualenv
virtualenv -p python3.9 venv
source ${APP_ROOT}/venv/bin/activate   # important
```

* Clone HERE_website github repository
```bash
## clone HERE_website
if [ -d ${WEB_ROOT} ]; then rm -rf ${WEB_ROOT}; fi
if [ -e ${APP_ROOT}/HERE_website.zip ]; then
   unzip ${APP_ROOT}/HERE_website.zip -d $WEB_ROOT;
else
   git clone https://github.com/data2intelligence/HERE_website.git ${WEB_ROOT}
fi
```

* Install Python libraries in virtual environment
```bash
# install python packages
cd ${WEB_ROOT} && pip3 install -r requirements.txt
```

* Install mod_wsgi module for Apache httpd server (skip if installed)
```bash
# Compile mod_wsgi (need root), skip if already installed
cd ${HERE_DEPS_TMP}
if [ ! -e mod_wsgi_5.0.0.tar.gz ]; then wget https://github.com/GrahamDumpleton/mod_wsgi/archive/refs/tags/5.0.0.tar.gz -O mod_wsgi_5.0.0.tar.gz; fi
if [ -d mod_wsgi_5.0.0 ]; then rm -rf mod_wsgi_5.0.0; fi
tar -xvf mod_wsgi_5.0.0.tar.gz
cd mod_wsgi-5.0.0
./configure --with-python=`which python3`
make
ls -alth ./src/server/.libs/
sudo make install
sudo chmod 755 /usr/lib64/httpd/modules/mod_wsgi.so	# important
```

* Modify Apache httpd server configuration according to the environmental variables
```bash
# modify /etc/httpd/conf/httpd.conf
echo """
###################################
#        HERE configuration
###################################
<VirtualHost *:80>
   ServerAlias HERE
   ErrorLog /var/log/httpd/error_log
   CustomLog /var/log/httpd/access_log combined
   TimeOut 600


   <Directory ${WEB_ROOT}>
       Require all granted
   </Directory>


   WSGIDaemonProcess HERE user=zhongz2 group=zhongz2 threads=2 home=${WEB_ROOT} python-path=${WEB_ROOT}:${APP_ROOT}/venv/lib/python3.9/site-packages:${APP_ROOT}/deps/install/lib/python3.9/:${APP_ROOT}/deps/install/lib/python3.9/site-packages
   WSGIPassAuthorization On
   WSGIProcessGroup HERE
   WSGIApplicationGroup %{GLOBAL}
   WSGIScriptAlias / ${WEB_ROOT}/wsgi.py
   WSGIScriptReloading On


   <Directory ${WEB_ROOT}>
       Order deny,allow
       Require all granted
   </Directory>
</VirtualHost>
""" | sudo tee -a /etc/httpd/conf/httpd.conf
```

* Enable mod_wsgi module in Apache httpd server
```bash
# add the following to /etc/httpd/conf.modules.d/10-wsgi.conf
<IfModule !wsgi_module>
   LoadModule wsgi_module modules/mod_wsgi.so
</IfModule>
```


* Generate files according to the environment variables
```bash
# generate wsgi.py
cd ${WEB_ROOT}
cat > wsgi.py <<EOL
#!${APP_ROOT}/venv/bin/python
from app import app as application
EOL 

# make soft links from HERE_assets to the HERE website root
cd ${WEB_ROOT}
ln -sf ${DST_DATA_ROOT} data_HiDARE_PLIP_20240208
```


(2), Copy the website code from the Github repository
https://github.com/data2intelligence/HERE_website

(3), Copy files from NIH Helix node
All data saved in NIH Helix node, please run the following commands from the web server node

* Set environment variables
```bash
##########################################################################
# Directory structures:
#   /boot
#   /local/
#       NCI_HERE_app/
#           website/                
#           tmp/
#           install/
#   /mnt/
#       disks/
#           disk1/
#               HERE_assets/                    # Please make sure that the disk1 has 5TB+ free space
##########################################################################
APP_ROOT=/local/NCI_HERE_app                    # app root (CHANGE THIS)
WEB_ROOT=${APP_ROOT}/website                    # website root (CHANGE THIS)
HERE_DEPS_INSTALL=${APP_ROOT}/install           # dependencies installed
HERE_DEPS_TMP=${APP_ROOT}/tmp                   # dependencies source, build, etc.
DST_DATA_ROOT=/mnt/hidare-efs/data/HERE_assets      # data assets, e.g., images (3TB+ free space)
APP_USERNAME=zhongz2				  # CHANGE THIS
##########################################################################
# please make sure the operating user can read and write these directories on the web server (skip if already existed)
sudo mkdir -p ${DST_DATA_ROOT}/temp_images_dir ${APP_ROOT} ${HERE_DEPS_INSTALL} ${HERE_DEPS_TMP}
sudo chown ${APP_USERNAME}:${APP_USERNAME} -R ${APP_ROOT} ${HERE_DEPS_INSTALL} ${HERE_DEPS_TMP}  # optional
sudo chown ${APP_USERNAME}:${APP_USERNAME} ${DST_DATA_ROOT}						   # optional
export SRC_HOST=helix.nih.gov:
export DST_HOST=
```

* Copy data and decompress data
```bash
########################### copy data processing scripts to the web server ###################
rsync -avh \
   ${SRC_HOST}/data/Jiang_Lab/Data/Zisha_Zhong/temp_20240208_hidare/deployment_scripts \
   ${DST_HOST}${APP_ROOT}/


########################## NCI data ############################
# copy data
rsync -avhrv --exclude PanCancer2GPUsFP \
   ${SRC_HOST}/data/Jiang_Lab/Data/Zisha_Zhong/temp_20240208/differential_results/20240208v4_NCIData \
   ${DST_HOST}${DST_DATA_ROOT}/
# decompress data
bash ${APP_ROOT}/deployment_scripts/unzip_files.sh ${DST_DATA_ROOT}/20240208v4_NCIData/big_images/


########################## TCGA ############################
# copy data
mkdir -p ${DST_DATA_ROOT}/20240208v4_TCGA-COMBINED/big_images
names=("BRCA" "PAAD" "CHOL" "UCS" "DLBC" "UVM" "UCEC" "MESO" "ACC" "KICH" "THYM" "TGCT" "PCPG" "ESCA" "SARC" "CESC" "PRAD" "THCA" "OV" "KIRC" "BLCA" "STAD" "SKCM" "READ" "LUSC" "LUAD" "LIHC" "LGG" "KIRP" "HNSC" "COAD" "GBM")
names=("SARC" "CESC" "PRAD" "THCA" "OV" "KIRC" "BLCA" "STAD" "SKCM" "READ" "LUSC" "LUAD" "LIHC" "LGG" "KIRP" "HNSC" "COAD" "GBM")
names=("BRCA" "CHOL" "DLBC" "MESO" "BLCA")
for pi in ${!names[@]}; do
   rsync -avh \
       ${SRC_HOST}/data/Jiang_Lab/Data/Zisha_Zhong/temp_20240208/differential_results/20240208v4_TCGA_${names[${pi}]}/big_images/* \
       ${DST_HOST}${DST_DATA_ROOT}/20240208v4_TCGA-COMBINED/big_images/
   echo ${pi}
done
# decompress data
bash ${APP_ROOT}/deployment_scripts/unzip_files.sh ${DST_DATA_ROOT}/20240208v4_TCGA-COMBINED/big_images/




########################## ST data ############################
# copy data
rsync -avhrc \
   --exclude cached_data \
   --exclude save_root_gene_da_dir \
   --exclude big_images_heatmap \
   --exclude vst_dir \
   --exclude svs \
   ${SRC_HOST}/data/Jiang_Lab/Data/Zisha_Zhong/temp_20240202/differential_results/20240202v4_ST \
   ${DST_HOST}${DST_DATA_ROOT}/
# decompress data (two command lines)
bash ${APP_ROOT}/deployment_scripts/unzip_files.sh ${DST_DATA_ROOT}/20240202v4_ST/big_images/
bash ${APP_ROOT}/deployment_scripts/unzip_files.sh ${DST_DATA_ROOT}/20240202v4_ST/PanCancer2GPUsFP/shared_attention_imagenetPLIP/split1_e95_h224_density_vis/feat_before_attention_feat/test/patch_images/


########################## assets data ############################
# copy data
rsync -avh \
   ${SRC_HOST}/data/Jiang_Lab/Data/Zisha_Zhong/temp_20240208_hidare/assets_20240630.zip \
   ${DST_HOST}${DST_DATA_ROOT}/
rsync -avh \
   ${SRC_HOST}/data/Jiang_Lab/Data/Zisha_Zhong/temp_20240208_hidare/lmdb_images_20240622_64 \
   ${DST_HOST}${DST_DATA_ROOT}/
rsync -avh \
   ${SRC_HOST}/data/Jiang_Lab/Data/Zisha_Zhong/temp_20240208_hidare/HERE_20240702.sql \
   ${DST_HOST}${DST_DATA_ROOT}/
# decompress data if you are on Helix, or run the commands one by one
cd ${DST_DATA_ROOT}/;
unzip assets_20240630.zip;
```


(4), Establish PROD database from dump file
Please run the following command to import HERE database to the MySQL database
```bash
# create HERE database in MySQL
mysql -u root -p
mysql> create database hidare_app;
mysql> exit;
# import data
sudo mysql -p -u root hidare_app < ${DST_DATA_ROOT}/HERE_20240702.sql
```

(5), Start the server and HERE app
```bash
sudo setenforce 0                   # disable SELinux temporarily, or go to /etc/selinux/config, disable it forever
sudo systemctl restart httpd                  

```
* Check the /var/log/httpd/error_log for Apache httpd log














