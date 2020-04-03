"""
WSGI config for DjangoProj project.

It exposes the WSGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/2.2/howto/deployment/wsgi/
"""

# For when it is deployed in AWS
# import os
# import sys
# sys.path.append('/opt/bitnami/apps/django/django_projects/tutorial')
# os.environ.setdefault("PYTHON_EGG_CACHE", "/opt/bitnami/apps/django/django_projects/tutorial/egg_cache")
# os.environ.setdefault("DJANGO_SETTINGS_MODULE", "tutorial.settings")
# from django.core.wsgi import get_wsgi_application
# application = get_wsgi_application()


import os

from django.core.wsgi import get_wsgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'DjangoProj.settings')

application = get_wsgi_application()
