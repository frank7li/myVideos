from celery import Celery

celery = Celery('myVideos')
celery.config_from_object('config.Config') 
celery.autodiscover_tasks(['tasks'])

celery.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='Europe/Oslo',
    enable_utc=True,
)