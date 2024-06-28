ps auxww | grep 'celery worker' | awk '{print $2}' | xargs kill -9
sleep 1
echo "starting celery process, please wait ..."
celery --app app.celery worker --loglevel=info --logfile=celery.log --pool gevent --concurrency=2






