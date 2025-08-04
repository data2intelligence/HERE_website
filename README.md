
# Web engine for tumor pathology image retrievals on massive scales

**HERE development code: https://github.com/data2intelligence/HERE_training**

**HERE website deployment code: https://github.com/data2intelligence/HERE_website**

**HERE website app: https://hereapp.ccr.cancer.gov**

**HERE101 benchmark dataset: https://github.com/data2intelligence/HERE101**

**[Jiang Lab, CDSL, CCR, NCI, NIH](https://ccr.cancer.gov/staff-directory/peng-jiang)**

[Zisha Zhong](https://scholar.google.com/citations?user=FYvMdNcAAAAJ), [Jinlin Huang], [Xinjing Li], [Lijuan Song], [Lanqi Gong], [Beibei Ru](https://scholar.google.com/citations?user=QB7Aj4YAAAAJ&hl=en), [Rohit Paul], [Jason Levine], [Yu Zhang], [Kenneth Aldape](https://ccr.cancer.gov/staff-directory/kenneth-aldape), [Peng Jiang](https://ccr.cancer.gov/staff-directory/peng-jiang)

[[`Paper`](https://ccr.cancer.gov/staff-directory/peng-jiang)] [[`Project`](https://ccr.cancer.gov/staff-directory/peng-jiang)] [[`HERE App`](https://hereapp.ccr.cancer.gov/)] [[`HERE101 Dataset`](https://github.com/data2intelligence/HERE101)] [[`BibTeX`](#citing-here-paper)]

# Abstract

Hematoxylin and Eosin staining (H&E) is widely used in clinical practice, but efficient and versatile Google-like image retrieval tools are lacking. We developed the H&E Retrieval Engine (HERE, https://hereapp.ccr.cancer.gov) to analyze patient cases based on image similarities to database records. Using H&E image regions as input, HERE searches 21.2 terabytes of whole-slide images from multiple tumor histopathology cohorts through a 12.1-gigabyte memory index, and returns top images containing regions similar to the query. HERE scans high-resolution images in the database using accurate artificial intelligence encoding and ultra-efficient hierarchical skip indexing. HERE demonstrated performance superior to existing image retrieval tools based on blinded pathologist scoring using benchmark queries that represent key image features of human tumors. By pairing spatial transcriptomics with H&E images, HERE also enables retrieving image features from gene transcriptomics input and identifies molecular pathways associated with tumor histologies.

![HERE](Fig1.jpg?raw=true)

# Setting up the HERE server 

To deploy the HERE website on a Linux server, we need to install a web server and the necessary libraries and softwares (e.g., Apache Httpd server, MySQL database, Python environment, etc.). If you're unfamiliar with these concepts, please search on Google or ask ChatGPT for more information.

* [1. Prerequisite](#sec1)
* [2. Apache server configuration](#sec2)
* [3. Data transfer](#sec3)
* [4. Server and database setup](#sec4)

<h2 id="sec1">1. Prerequisite</h2>

1.1 Directory structure and set the common environmental variables

First, we need to create some directories and set environment variables on the server. You can run the following commands one by one in your Bash terminal. Make sure you understand the purpose of each line and modify them as needed based on your server configuration.
Lines starting with `#` are comments and can be ignored.
Root privileges are required to set up the server.

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
APP_ROOT=/local/NCI_HERE_app                       # app root (CHANGE THIS)
WEB_ROOT=${APP_ROOT}/website                       # website root 
HERE_DEPS_INSTALL=${APP_ROOT}/install              # dependencies installed
HERE_DEPS_TMP=${APP_ROOT}/tmp                      # dependencies source, build, etc.
DST_DATA_ROOT=/mnt/hidare-efs/data/HERE_assets     # data assets, e.g., images (3TB+ free space)
APP_USERNAME=zhongz2				                     # CHANGE THIS
##########################################################################
# please make sure the operating user can read and write these directories on the web server (skip if already existed)
sudo mkdir -p ${DST_DATA_ROOT}/temp_images_dir ${DST_DATA_ROOT}/assets/ST_kmeans_clustering ${APP_ROOT} ${HERE_DEPS_INSTALL} ${HERE_DEPS_TMP}
sudo chown ${APP_USERNAME}:${APP_USERNAME} -R ${APP_ROOT} ${HERE_DEPS_INSTALL} ${HERE_DEPS_TMP}
sudo chown ${APP_USERNAME}:${APP_USERNAME} -R ${DST_DATA_ROOT}						   

export PATH=${HERE_DEPS_INSTALL}/bin:$PATH
export LD_LIBRARY_PATH=${HERE_DEPS_INSTALL}/lib64:${HERE_DEPS_INSTALL}/lib:$LD_LIBRARY_PATH
export PKG_CONFIG_PATH=${HERE_DEPS_INSTALL}/lib64/pkgconfig:${HERE_DEPS_INSTALL}/lib/pkgconfig:$PKG_CONFIG_PATH
```

1.2 Add the following environmental variables into your bash profile (~/.bashrc)

```
export HERE_DB_USER="hidare_app"
export HERE_DB_PASSWORD="your password"           # change this
export HERE_DB_HOST="your mysql database host"    # change this
export HERE_DB_DATABASE="hidare"
```


1.3 Install dependency libraries on CentOS 7

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
```

1.4 Install OpenSSL 1.1.1

```bash
# ImportError: urllib3 v2 only supports OpenSSL 1.1.1+, currently the 'ssl' module is compiled with 'OpenSSL 1.0.2k-fips  26 Jan 2017'. See: https://github.com/urllib3/urllib3/issues/2168

cd $HERE_DEPS_TMP
wget https://github.com/openssl/openssl/releases/download/OpenSSL_1_1_1s/openssl-1.1.1s.tar.gz
tar -xvf openssl-1.1.1s.tar.gz
cd openssl-1.1.1s
./config --prefix=${HERE_DEPS_INSTALL} --openssldir=${HERE_DEPS_INSTALL}/etc/ssl --libdir=lib no-shared zlib-dynamic
make
make install
```

1.5 Install MySQL
```bash
cd $HERE_DEPS_TMP
curl -sSLO https://dev.mysql.com/get/mysql84-community-release-el7-1.noarch.rpm
md5sum mysql84-community-release-el7-1.noarch.rpm
sudo rpm -ivh mysql84-community-release-el7-1.noarch.rpm
sudo yum install -y mysql-server
sudo systemctl start mysqld
```

1.6 Install Python-3.9.18
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

1.7 Install Python virtual environment
```bash
# Virtual environment
cd ${APP_ROOT}
pip3 install virtualenv
virtualenv -p python3.9 venv
source ${APP_ROOT}/venv/bin/activate   # important
```

1.8 Clone HERE_website github repository
```bash
## clone HERE_website
if [ -d ${WEB_ROOT} ]; then rm -rf ${WEB_ROOT}; fi
if [ -e ${APP_ROOT}/HERE_website.zip ]; then
   unzip ${APP_ROOT}/HERE_website.zip -d $WEB_ROOT;
else
   git clone https://github.com/data2intelligence/HERE_website.git ${WEB_ROOT}
fi
```

1.9 Install Python libraries in virtual environment
```bash
# install python packages
cd ${WEB_ROOT} && pip3 install -r requirements.txt
```


<h2 id="sec2">2. Apache server configuration</h2>



2.1 Install mod_wsgi module for Apache httpd server
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

2.2 Modify Apache httpd server configuration according to the environmental variables
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

2.3 Enable mod_wsgi module in Apache httpd server
```bash
# add the following to /etc/httpd/conf.modules.d/10-wsgi.conf
<IfModule !wsgi_module>
   LoadModule wsgi_module modules/mod_wsgi.so
</IfModule>
```


2.4 Generate files according to the environment variables
```bash
# generate wsgi.py
cd ${WEB_ROOT}
cat > wsgi.py << EOL
#!${APP_ROOT}/venv/bin/python
import os
os.environ['HERE_DB_USER'] = "hidare_app"
os.environ['HERE_DB_PASSWORD'] = "your password"  # change this
os.environ['HERE_DB_HOST'] = "your server"    # change this
os.environ['HERE_DB_DATABASE'] = "hidare"
from app import app as application
EOL


# make soft links from HERE_assets to the HERE website root
cd ${WEB_ROOT}
ln -sf ${DST_DATA_ROOT} data_HiDARE_PLIP_20240208
```

<h2 id="sec3">3. Data Transfer</h2>


3.1 Set environment variables in case you don't set them
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
sudo mkdir -p ${DST_DATA_ROOT}/temp_images_dir ${DST_DATA_ROOT}/assets/ST_kmeans_clustering ${APP_ROOT} ${HERE_DEPS_INSTALL} ${HERE_DEPS_TMP}
sudo chown ${APP_USERNAME}:${APP_USERNAME} -R ${APP_ROOT} ${HERE_DEPS_INSTALL} ${HERE_DEPS_TMP}  
sudo chown ${APP_USERNAME}:${APP_USERNAME} -R ${DST_DATA_ROOT}

export SRC_HOST=helix.nih.gov:
export DST_HOST=
```

3.2 Copy data and decompress data

```bash
# copy data processing scripts to the web server
rsync -avh \
   ${SRC_HOST}/data/Jiang_Lab/Data/Zisha_Zhong/temp_20240208_hidare/deployment_scripts \
   ${DST_HOST}${APP_ROOT}/


# copy assets data 
rsync -avhrv \
   ${SRC_HOST}/data/Jiang_Lab/Data/Zisha_Zhong/HERE_website_assets/* \
   ${DST_HOST}${DST_DATA_ROOT}/

# decompress data (this will take a few days)
bash ${APP_ROOT}/deployment_scripts/unzip_files.sh ${DST_DATA_ROOT}/NCIData_big_images/
bash ${APP_ROOT}/deployment_scripts/unzip_files.sh ${DST_DATA_ROOT}/TCGA-COMBINED_big_images/
bash ${APP_ROOT}/deployment_scripts/unzip_files.sh ${DST_DATA_ROOT}/CPTAC_big_images/
tar -xvf ${DST_DATA_ROOT}/assets_20250720.tar.gz -C ${DST_DATA_ROOT}/assets/
tar -xvf ${DST_DATA_ROOT}/assets_20250721_models.tar.gz -C ${DST_DATA_ROOT}/assets/
unzip ${DST_DATA_ROOT}/ST_processed_data.zip -d ${DST_DATA_ROOT}/assets/ST_kmeans_clustering/
ln -sf ${DST_DATA_ROOT}/assets/ST_kmeans_clustering/big_images ${DST_DATA_ROOT}/ST_big_images
```

<h2 id="sec4">4. Server and database setup</h2>


4.1 Setup the MySQL server

Please run the following command to import HERE database to the MySQL database.
```bash
# create HERE database in MySQL
mysql -u root -p
mysql> create database ${HERE_DB_DATABASE};
mysql> exit;

# import data
cd ${WEB_ROOT}
python add_data_to_mysql.py  

# mysql --host=${HERE_DB_HOST} --user=${HERE_DB_USER} --password=${HERE_DB_PASSWORD} ${HERE_DB_DATABASE}
```

4.2 Start the web server

Please run the following command to start HERE web app on the server.

```bash
sudo setenforce 0                   # disable SELinux temporarily, or go to /etc/selinux/config, disable it forever
sudo systemctl restart httpd                  
```
and then check the /var/log/httpd/error_log for Apache httpd log. Then you can open the website in your web browser.


## Citing HERE

If you use HERE or HERE101 dataset in your research, please use the following BibTeX entry.

```bibtex
@article{jiang2024ncihereapp,
  title={Web engine for tumor pathology image retrievals on massive scales},
  author={Zhong, Zisha and Huang, Jinlin and Li, Xinjing, and Song, Lijuan and Gong, Lanqi and Ru, Beibei and Paul, Rohit and Levine, Jason and Zhang, Yu and Aldape, Kenneth and Jiang Peng},
  journal={Submitted},
  url={Submitted},
  year={2024}
}
```

