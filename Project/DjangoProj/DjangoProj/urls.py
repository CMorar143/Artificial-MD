"""DjangoProj URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""

from django.contrib import admin
from django.urls import path, include
from django.conf.urls import url
from django.views.generic.base import TemplateView
# from django.contrib.staticfiles.urls import staticfiles_urlpatterns

from boards import views
from boards.views import exam, patient, patient_info, home, reminders, diary

urlpatterns = [
    path('admin/', admin.site.urls),
    path('accounts/', include('django.contrib.auth.urls')),
    path('', home.as_view(), name='home'),
    path('exam/', exam.as_view(), name='exam'),
    path('patient/', patient.as_view(), name='patient'),
    path('reminders/', reminders.as_view(), name='reminders'),
    path('diary/', diary.as_view(), name='diary'),
    path('patient_info/<int:id>/', patient_info.as_view(), name='patient_info'),
    path('results/', views.results.as_view(), name='results')
]
